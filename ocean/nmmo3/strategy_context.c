#include "strategy_context.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int abs_int(int value) {
    return value < 0 ? -value : value;
}

static int distance_between(int r1, int c1, int r2, int c2) {
    return abs_int(r1 - r2) + abs_int(c1 - c2);
}

static int is_local(MMO* env, Entity* self, Entity* other) {
    return abs_int(self->r - other->r) <= env->y_window &&
           abs_int(self->c - other->c) <= env->x_window;
}

static void fill_entity(StrategyEntity* dst, Entity* src, Entity* self, int id) {
    dst->id = id;
    dst->rel_r = src->r - self->r;
    dst->rel_c = src->c - self->c;
    dst->distance = distance_between(self->r, self->c, src->r, src->c);
    dst->comb_lvl = src->comb_lvl;
    dst->prof_lvl = src->prof_lvl;
    dst->hp = src->hp;
    dst->hp_max = src->hp_max;
    dst->element = src->element;
    dst->in_combat = src->in_combat;
    dst->ranged = src->ranged;
    dst->equipment_attack = src->equipment_attack;
    dst->equipment_defense = src->equipment_defense;
    dst->goal = src->goal;
}

static int append_text(char* text, int offset, const char* format, ...) {
    va_list args;
    int written;
    if (offset >= STRATEGY_CONTEXT_MAX - 1) {
        return STRATEGY_CONTEXT_MAX;
    }
    va_start(args, format);
    written = vsnprintf(text + offset, STRATEGY_CONTEXT_MAX - offset, format, args);
    va_end(args);
    if (written < 0) {
        return -1;
    }
    if (written >= STRATEGY_CONTEXT_MAX - offset) {
        return STRATEGY_CONTEXT_MAX;
    }
    return offset + written;
}

static const char* action_name(int action) {
    static const char* names[26] = {
        "MOVE_DOWN", "MOVE_UP", "MOVE_RIGHT", "MOVE_LEFT", "NOOP",
        "ATTACK", "INVALID", "UI", "USE_ITEM_1", "USE_ITEM_2",
        "USE_ITEM_3", "USE_ITEM_4", "USE_ITEM_5", "USE_ITEM_6",
        "USE_ITEM_7", "USE_ITEM_8", "USE_ITEM_9", "USE_ITEM_0",
        "USE_ITEM_MINUS", "USE_ITEM_EQUALS", "BUY", "SELL",
        "MOVE_DOWN_SHIFT", "MOVE_UP_SHIFT", "MOVE_RIGHT_SHIFT",
        "MOVE_LEFT_SHIFT"
    };
    return action >= 0 && action < 26 ? names[action] : "INVALID";
}

static void record_qwen3_action(Log* log, int action) {
    if (action == ATN_NOOP) {
        log->qwen3_noop_actions += 1;
    } else if (action == ATN_ATTACK) {
        log->qwen3_attack_actions += 1;
    } else if (is_move(action)) {
        log->qwen3_move_actions += 1;
    } else if (is_run(action)) {
        log->qwen3_shift_actions += 1;
    } else if (is_num(action)) {
        log->qwen3_item_actions += 1;
    } else if (action == ATN_BUY || action == ATN_SELL) {
        log->qwen3_market_actions += 1;
    }
}

int build_strategy_context(MMO* env, int pid, StrategyContext* context) {
    int offset = 0;
    int group_alive = 0;
    int group_dead = 0;
    int group_r = 0;
    int group_c = 0;
    int group_comb = 0;
    int group_prof = 0;
    int group_hp = 0;
    int group_hp_max = 0;
    int group_gold = 0;
    Entity* self;
    Reward* reward;

    if (env == NULL || context == NULL || pid < 0 || pid >= env->num_agents ||
        env->players == NULL || env->items == NULL || env->reward_struct == NULL) {
        return -1;
    }

    memset(context, 0, sizeof(*context));
    context->pid = pid;
    self = &env->players[pid];
    reward = &env->reward_struct[pid];

    context->self_r = self->r;
    context->self_c = self->c;
    context->self_comb_lvl = self->comb_lvl;
    context->self_prof_lvl = self->prof_lvl;
    context->self_hp = self->hp;
    context->self_hp_max = self->hp_max;
    context->self_gold = self->gold;
    context->self_in_combat = self->in_combat;
    context->self_equipment_attack = self->equipment_attack;
    context->self_equipment_defense = self->equipment_defense;
    context->self_wander_range = self->wander_range;
    context->self_ranged = self->ranged;
    context->self_goal = self->goal;
    memcpy(context->equipment, self->equipment, sizeof(context->equipment));
    memcpy(context->inventory, self->inventory, sizeof(context->inventory));
    memcpy(context->is_equipped, self->is_equipped, sizeof(context->is_equipped));

    context->reward_death = reward->death;
    context->reward_pioneer = reward->pioneer;
    context->reward_comb_lvl = reward->comb_lvl;
    context->reward_prof_lvl = reward->prof_lvl;
    context->reward_item_atk_lvl = reward->item_atk_lvl;
    context->reward_item_def_lvl = reward->item_def_lvl;
    context->reward_item_tool_lvl = reward->item_tool_lvl;
    context->reward_market_buy = reward->market_buy;
    context->reward_market_sell = reward->market_sell;

    for (int i = 0; i < env->num_agents && context->num_players < STRATEGY_CONTEXT_MAX_ENTITIES; i++) {
        if (i != pid && is_local(env, self, &env->players[i])) {
            fill_entity(&context->players[context->num_players++], &env->players[i], self, i);
        }
    }
    for (int i = 0; i < env->num_enemies && context->num_enemies < STRATEGY_CONTEXT_MAX_ENTITIES; i++) {
        if (is_local(env, self, &env->enemies[i])) {
            fill_entity(&context->enemies[context->num_enemies++], &env->enemies[i], self, i);
        }
    }

    for (int r = self->r - env->y_window; r <= self->r + env->y_window; r++) {
        if (r < 0 || r >= env->height) continue;
        for (int c = self->c - env->x_window; c <= self->c + env->x_window; c++) {
            int item_id;
            StrategyResource* resource;
            if (c < 0 || c >= env->width) continue;
            item_id = env->items[r * env->width + c];
            if (item_id == I_NULL) continue;
            if (context->num_resources >= STRATEGY_CONTEXT_MAX_RESOURCES) break;
            resource = &context->resources[context->num_resources++];
            resource->item_id = item_id;
            if (item_id >= 0 && item_id < (MAX_TIERS + 1) * I_N + 1) {
                resource->type = ITEMS[item_id].type;
                resource->tier = ITEMS[item_id].tier;
            }
            resource->rel_r = r - self->r;
            resource->rel_c = c - self->c;
            resource->distance = distance_between(self->r, self->c, r, c);
        }
        if (context->num_resources >= STRATEGY_CONTEXT_MAX_RESOURCES) break;
    }

    context->market_buys = env->market_buys;
    context->market_sells = env->market_sells;

    for (int i = 0; i < env->num_agents; i++) {
        Entity* player = &env->players[i];
        group_r += player->r;
        group_c += player->c;
        group_comb += player->comb_lvl;
        group_prof += player->prof_lvl;
        group_hp += player->hp;
        group_hp_max += player->hp_max;
        group_gold += player->gold;
        if (player->hp > 0) {
            group_alive++;
        } else {
            group_dead++;
        }
    }
    if (env->num_agents > 0) {
        group_r /= env->num_agents;
        group_c /= env->num_agents;
        group_comb /= env->num_agents;
        group_prof /= env->num_agents;
        group_hp /= env->num_agents;
        group_hp_max /= env->num_agents;
        group_gold /= env->num_agents;
    }

    offset = append_text(context->text, offset,
        "NMMO_STRATEGY_CONTEXT\nstrategy_scope=GROUP\ncontroller_agent=%d\ntick=%d\n\n"
        "GROUP_STATE\nagent_count=%d\nalive_agents=%d\ndead_agents=%d\n"
        "average_position=%d,%d\naverage_combat_level=%d\n"
        "average_profession_level=%d\naverage_hp=%d/%d\naverage_gold=%d\n\n"
        "SELF\n"
        "position=%d,%d\ncombat_level=%d\nprofession_level=%d\nhp=%d/%d\n"
        "gold=%d\nin_combat=%d\nequipment_attack=%d\nequipment_defense=%d\n"
        "wander_range=%d\nranged=%d\ngoal=%d\n\n",
        pid, env->tick, env->num_agents, group_alive, group_dead,
        group_r, group_c, group_comb, group_prof, group_hp, group_hp_max, group_gold,
        context->self_r, context->self_c, context->self_comb_lvl,
        context->self_prof_lvl, context->self_hp, context->self_hp_max, context->self_gold,
        context->self_in_combat, context->self_equipment_attack,
        context->self_equipment_defense, context->self_wander_range,
        context->self_ranged, context->self_goal);
    offset = append_text(context->text, offset, "EQUIPMENT slots=");
    for (int i = 0; i < 5; i++) offset = append_text(context->text, offset, "%d%s", context->equipment[i], i == 4 ? "\n" : ",");
    offset = append_text(context->text, offset, "INVENTORY items=");
    for (int i = 0; i < INVENTORY_SIZE; i++) offset = append_text(context->text, offset, "%d%s", context->inventory[i], i == INVENTORY_SIZE - 1 ? "\n" : ",");
    offset = append_text(context->text, offset,
        "\nPROGRESSION\nreward_death=%.2f\nreward_pioneer=%.2f\nreward_combat=%.2f\n"
        "reward_profession=%.2f\nreward_item_attack=%.2f\nreward_item_defense=%.2f\n"
        "reward_item_tool=%.2f\nreward_market_buy=%.2f\nreward_market_sell=%.2f\n",
        context->reward_death, context->reward_pioneer, context->reward_comb_lvl,
        context->reward_prof_lvl, context->reward_item_atk_lvl, context->reward_item_def_lvl,
        context->reward_item_tool_lvl, context->reward_market_buy, context->reward_market_sell);

    offset = append_text(context->text, offset, "\nNEARBY_PLAYERS count=%d\n", context->num_players);
    for (int i = 0; i < context->num_players; i++) {
        StrategyEntity* e = &context->players[i];
        offset = append_text(context->text, offset, "player id=%d rel=(%d,%d) dist=%d combat=%d prof=%d hp=%d/%d element=%d combat_state=%d ranged=%d atk=%d def=%d goal=%d\n", e->id, e->rel_r, e->rel_c, e->distance, e->comb_lvl, e->prof_lvl, e->hp, e->hp_max, e->element, e->in_combat, e->ranged, e->equipment_attack, e->equipment_defense, e->goal);
    }
    offset = append_text(context->text, offset, "\nNEARBY_ENEMIES count=%d\n", context->num_enemies);
    for (int i = 0; i < context->num_enemies; i++) {
        StrategyEntity* e = &context->enemies[i];
        offset = append_text(context->text, offset, "enemy id=%d rel=(%d,%d) dist=%d combat=%d hp=%d/%d element=%d combat_state=%d ranged=%d atk=%d def=%d goal=%d\n", e->id, e->rel_r, e->rel_c, e->distance, e->comb_lvl, e->hp, e->hp_max, e->element, e->in_combat, e->ranged, e->equipment_attack, e->equipment_defense, e->goal);
    }
    offset = append_text(context->text, offset, "\nNEARBY_ITEMS count=%d\n", context->num_resources);
    for (int i = 0; i < context->num_resources; i++) {
        StrategyResource* r = &context->resources[i];
        offset = append_text(context->text, offset, "item id=%d type=%d tier=%d rel=(%d,%d) dist=%d\n", r->item_id, r->type, r->tier, r->rel_r, r->rel_c, r->distance);
    }
    offset = append_text(context->text, offset, "\nRECENT_STRATEGIES count=%d\n", env->qwen3_history_count);
    for (int i = 0; i < env->qwen3_history_count; i++) {
        int history_idx = (env->qwen3_history_next - env->qwen3_history_count + i + QWEN3_HISTORY_SIZE) % QWEN3_HISTORY_SIZE;
        offset = append_text(context->text, offset, "%s\n", env->qwen3_history[history_idx]);
    }
    offset = append_text(context->text, offset, "\nMARKET\nbuys=%d\nsells=%d\n", context->market_buys, context->market_sells);
    if (offset < 0) return -1;
    context->text[STRATEGY_CONTEXT_MAX - 1] = '\0';
    return offset >= STRATEGY_CONTEXT_MAX ? STRATEGY_CONTEXT_MAX - 1 : offset;
}

int nmmo3_qwen3_action(MMO* env, int pid) {
    int input_pipe[2];
    int output_pipe[2];
    pid_t child;
    StrategyContext context;
    char response[2048];
    ssize_t bytes_read;
    int status;
    int action = env->qwen3_current_action;
    const int qwen3_timeout_seconds = 15;

    #define RECORD_QWEN3_HISTORY(result) do { \
        snprintf(env->qwen3_history[env->qwen3_history_next], QWEN3_HISTORY_ENTRY_SIZE, \
            "tick=%d action=%s (%d) result=%s", env->tick, action_name(action), action, result); \
        env->qwen3_history_next = (env->qwen3_history_next + 1) % QWEN3_HISTORY_SIZE; \
        if (env->qwen3_history_count < QWEN3_HISTORY_SIZE) env->qwen3_history_count += 1; \
    } while (0)

    env->log.qwen3_decisions += 1;

    if (build_strategy_context(env, pid, &context) < 0 ||
        pipe(input_pipe) < 0 || pipe(output_pipe) < 0) {
        return action;
    }
    child = fork();
    if (child == 0) {
        dup2(input_pipe[0], STDIN_FILENO);
        dup2(output_pipe[1], STDOUT_FILENO);
        close(input_pipe[0]);
        close(input_pipe[1]);
        close(output_pipe[0]);
        close(output_pipe[1]);
        execlp("python3", "python3", "ocean/nmmo3/llm_strategy.py", (char*)NULL);
        _exit(127);
    }
    if (child < 0) {
        close(input_pipe[0]);
        close(input_pipe[1]);
        close(output_pipe[0]);
        close(output_pipe[1]);
        return action;
    }
    close(input_pipe[0]);
    close(output_pipe[1]);
    (void)write(input_pipe[1], context.text, strlen(context.text));
    (void)write(input_pipe[1], "\0", 1);
    close(input_pipe[1]);
    {
        fd_set read_fds;
        struct timeval timeout = {.tv_sec = qwen3_timeout_seconds, .tv_usec = 0};
        FD_ZERO(&read_fds);
        FD_SET(output_pipe[0], &read_fds);
        if (select(output_pipe[0] + 1, &read_fds, NULL, NULL, &timeout) <= 0) {
            kill(child, SIGKILL);
            close(output_pipe[0]);
            waitpid(child, &status, 0);
            env->log.qwen3_failures += 1;
            record_qwen3_action(&env->log, action);
            RECORD_QWEN3_HISTORY("timeout");
            return action;
        }
        bytes_read = read(output_pipe[0], response, sizeof(response) - 1);
    }
    close(output_pipe[0]);
    waitpid(child, &status, 0);
    if (bytes_read <= 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        env->log.qwen3_failures += 1;
        record_qwen3_action(&env->log, action);
        printf("Qwen3 unavailable; keeping previous group action: %s (%d)\n",
            action_name(action), action);
        RECORD_QWEN3_HISTORY("failed");
        return action;
    }
    response[bytes_read] = '\0';
    {
        char* action_field = strstr(response, "\"action_code\"");
        if (action_field) {
            (void)sscanf(action_field, "\"action_code\"%*[^0-9]%d", &action);
        }
    }
    if (action < ATN_DOWN || action > ATN_LEFT_SHIFT || action == 6) {
        action = env->qwen3_current_action;
    }
    record_qwen3_action(&env->log, action);
    RECORD_QWEN3_HISTORY("completed");
    printf("Qwen3 recommended group action: tick=%d agents=%d action=%s (%d)\n",
        env->tick, env->num_agents, action_name(action), action);
    #undef RECORD_QWEN3_HISTORY
    return action;
}
