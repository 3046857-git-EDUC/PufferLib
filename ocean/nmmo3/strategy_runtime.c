#include "strategy_context.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_STRATEGY_HORIZON 720

static atomic_flag strategy_lock = ATOMIC_FLAG_INIT;
static MMO* strategy_active_env;
static float strategy_preferences[9] = {1.0f};
static int strategy_expire_tick;

static void lock_strategy(void) {
    while (atomic_flag_test_and_set_explicit(&strategy_lock, memory_order_acquire)) {}
}

static void unlock_strategy(void) {
    atomic_flag_clear_explicit(&strategy_lock, memory_order_release);
}

void nmmo3_strategy_init(MMO* env) {
    lock_strategy();
    env->strategy_pid = -1;
    env->strategy_output_fd = -1;
    env->strategy_response_bytes = 0;
    memset(strategy_preferences, 0, sizeof(strategy_preferences));
    strategy_preferences[0] = 1.0f;
    strategy_expire_tick = 0;
    memcpy(env->hal_preferences, strategy_preferences, sizeof(strategy_preferences));
    unlock_strategy();
}

static int is_local(MMO* env, Entity* self, Entity* other) {
    return abs(self->r - other->r) <= env->y_window &&
        abs(self->c - other->c) <= env->x_window;
}

int build_strategy_context(MMO* env, int pid, StrategyContext* context) {
    Entity* self;
    Reward* reward;
    int alive_agents = 0;
    int nearby_enemies = 0;
    int nearby_resources = 0;
    float health_fraction = 0.0f;
    float base_return;

    if (env == NULL || context == NULL || pid < 0 || pid >= env->num_agents ||
            env->players == NULL || env->items == NULL || env->reward_struct == NULL) {
        return -1;
    }

    self = &env->players[pid];
    reward = &env->reward_struct[pid];
    for (int i = 0; i < env->num_agents; i++) {
        Entity* player = &env->players[i];
        if (player->hp > 0) alive_agents++;
        if (player->hp_max > 0) {
            health_fraction += (float)player->hp / player->hp_max;
        }
    }
    if (env->num_agents > 0) health_fraction /= env->num_agents;

    for (int i = 0; i < env->num_enemies; i++) {
        if (is_local(env, self, &env->enemies[i])) nearby_enemies++;
    }
    for (int r = self->r - env->y_window; r <= self->r + env->y_window; r++) {
        if (r < 0 || r >= env->height) continue;
        for (int c = self->c - env->x_window; c <= self->c + env->x_window; c++) {
            if (c >= 0 && c < env->width &&
                    env->items[r * env->width + c] != I_NULL) {
                nearby_resources++;
            }
        }
    }

    base_return = reward->death + reward->comb_lvl + reward->prof_lvl +
        reward->item_atk_lvl + reward->item_def_lvl +
        reward->market_buy + reward->market_sell;
    memset(context, 0, sizeof(*context));
    context->pid = pid;
    int written = snprintf(context->text, sizeof(context->text),
        "{\"tick\":%d,\"num_agents\":%d,\"alive_agents\":%d,"
        "\"mean_health_fraction\":%.9g,\"nearby_enemy_count\":%d,"
        "\"nearby_resource_count\":%d,\"recent_base_return\":%.9g,"
        "\"current_objective\":\"survive\",\"recent_events\":{"
        "\"death\":%.9g,\"pioneer\":%.9g,\"combat_level\":%.9g,"
        "\"profession_level\":%.9g,\"attack_equipment\":%.9g,"
        "\"defense_equipment\":%.9g,\"tool_equipment\":%.9g,"
        "\"market_purchase\":%.9g,\"market_sale\":%.9g}}",
        env->tick, env->num_agents, alive_agents, health_fraction,
        nearby_enemies, nearby_resources, base_return,
        reward->death, reward->pioneer, reward->comb_lvl, reward->prof_lvl,
        reward->item_atk_lvl, reward->item_def_lvl, reward->item_tool_lvl,
        reward->market_buy, reward->market_sell);
    if (written < 0 || written >= (int)sizeof(context->text)) return -1;
    return written;
}

static int parse_float_field(const char* object, const char* key, float* value) {
    char pattern[96];
    const char* field;
    const char* object_end = strchr(object, '}');
    char* value_end;
    double parsed;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    field = strstr(object, pattern);
    if (field == NULL || object_end == NULL || field > object_end) return 0;
    field += strlen(pattern);
    while (isspace((unsigned char)*field)) field++;
    if (*field++ != ':') return 0;
    while (isspace((unsigned char)*field)) field++;
    errno = 0;
    parsed = strtod(field, &value_end);
    if (value_end == field || errno == ERANGE || !isfinite(parsed)) return 0;
    while (isspace((unsigned char)*value_end)) value_end++;
    if (*value_end != ',' && *value_end != '}') return 0;
    *value = (float)parsed;
    return isfinite(*value);
}

static int parse_nonnegative_int(const char* response, const char* key, int* value) {
    char pattern[96];
    const char* field;
    char* value_end;
    long parsed;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    field = strstr(response, pattern);
    if (field == NULL) return 0;
    field += strlen(pattern);
    while (isspace((unsigned char)*field)) field++;
    if (*field++ != ':') return 0;
    while (isspace((unsigned char)*field)) field++;
    errno = 0;
    parsed = strtol(field, &value_end, 10);
    if (value_end == field || errno == ERANGE || parsed < 0 || parsed > INT_MAX) return 0;
    while (isspace((unsigned char)*value_end)) value_end++;
    if (*value_end != ',' && *value_end != '}') return 0;
    *value = (int)parsed;
    return 1;
}

static int valid_preferences(const float preferences[9]) {
    for (int i = 0; i < 9; i++) {
        if (!isfinite(preferences[i]) || preferences[i] < 0.0f ||
                preferences[i] > 1.0f) return 0;
    }
    return 1;
}

static int parse_guidance(const char* response, float preferences[9], int* horizon) {
    const char* field = strstr(response, "\"reward_preferences\"");
    const char* object = field == NULL ? NULL : strchr(field, '{');

    if (object == NULL ||
            !parse_float_field(object, "death", &preferences[0]) ||
            !parse_float_field(object, "pioneer", &preferences[1]) ||
            !parse_float_field(object, "combat_level", &preferences[2]) ||
            !parse_float_field(object, "profession_level", &preferences[3]) ||
            !parse_float_field(object, "attack_equipment", &preferences[4]) ||
            !parse_float_field(object, "defense_equipment", &preferences[5]) ||
            !parse_float_field(object, "tool_equipment", &preferences[6]) ||
            !parse_float_field(object, "market_purchase", &preferences[7]) ||
            !parse_float_field(object, "market_sale", &preferences[8]) ||
            !valid_preferences(preferences) ||
            !parse_nonnegative_int(response, "horizon", horizon) ||
            *horizon < 1 || *horizon > MAX_STRATEGY_HORIZON) {
        return 0;
    }
    return 1;
}

static void apply_strategy_preferences(MMO* env) {
    if (strategy_expire_tick > 0 && env->tick >= strategy_expire_tick) {
        memset(strategy_preferences, 0, sizeof(strategy_preferences));
        strategy_preferences[0] = 1.0f;
        strategy_expire_tick = 0;
    }
    memcpy(env->hal_preferences, strategy_preferences, sizeof(strategy_preferences));
}

static void record_strategy_failure(MMO* env, const char* message) {
    env->log.strategy_failures += 1.0f;
    fprintf(stderr, "NMMO3 strategy failed: %s\n", message);
}

static int launch_strategy(MMO* env, int pid) {
    int input_pipe[2];
    int output_pipe[2];
    pid_t child;
    StrategyContext context;
    int context_length;

    context_length = build_strategy_context(env, pid, &context);
    if (context_length < 0) {
        record_strategy_failure(env, "failed to build context");
        return 0;
    }
    if (pipe(input_pipe) < 0) {
        perror("NMMO3 strategy input pipe");
        record_strategy_failure(env, "could not create input pipe");
        return 0;
    }
    if (pipe(output_pipe) < 0) {
        close(input_pipe[0]);
        close(input_pipe[1]);
        perror("NMMO3 strategy output pipe");
        record_strategy_failure(env, "could not create output pipe");
        return 0;
    }

    child = fork();
    if (child == 0) {
        const char* python = getenv("NMMO3_STRATEGY_PYTHON");
        const char* script = getenv("NMMO3_STRATEGY_SCRIPT");
        if (python == NULL || python[0] == '\0') python = "python3";
        if (script == NULL || script[0] == '\0') {
            script = "ocean/nmmo3/ollama_strategy.py";
        }
        dup2(input_pipe[0], STDIN_FILENO);
        dup2(output_pipe[1], STDOUT_FILENO);
        close(input_pipe[0]);
        close(input_pipe[1]);
        close(output_pipe[0]);
        close(output_pipe[1]);
        execlp(python, python, script, (char*)NULL);
        _exit(127);
    }
    if (child < 0) {
        close(input_pipe[0]);
        close(input_pipe[1]);
        close(output_pipe[0]);
        close(output_pipe[1]);
        perror("NMMO3 strategy fork");
        record_strategy_failure(env, "could not start strategy process");
        return 0;
    }

    close(input_pipe[0]);
    close(output_pipe[1]);
    ssize_t written = write(input_pipe[1], context.text, (size_t)context_length);
    close(input_pipe[1]);
    if (written != context_length) {
        kill(child, SIGKILL);
        waitpid(child, NULL, 0);
        close(output_pipe[0]);
        record_strategy_failure(env, "could not send strategy context");
        return 0;
    }

    int flags = fcntl(output_pipe[0], F_GETFL);
    if (flags < 0 || fcntl(output_pipe[0], F_SETFL, flags | O_NONBLOCK) < 0) {
        kill(child, SIGKILL);
        waitpid(child, NULL, 0);
        close(output_pipe[0]);
        record_strategy_failure(env, "could not configure strategy output pipe");
        return 0;
    }

    env->log.strategy_decisions += 1.0f;
    env->strategy_pid = child;
    env->strategy_output_fd = output_pipe[0];
    env->strategy_response_bytes = 0;
    strategy_active_env = env;
    return 1;
}

int nmmo3_strategy_step(MMO* env, int pid) {
    int result = 0;

    lock_strategy();
    apply_strategy_preferences(env);

    if (env->strategy_pid > 0) {
        ssize_t bytes_read;
        while (env->strategy_response_bytes < (int)sizeof(env->strategy_response) - 1) {
            bytes_read = read(env->strategy_output_fd,
                env->strategy_response + env->strategy_response_bytes,
                sizeof(env->strategy_response) - 1 - (size_t)env->strategy_response_bytes);
            if (bytes_read > 0) {
                env->strategy_response_bytes += (int)bytes_read;
                continue;
            }
            if (bytes_read < 0 && errno == EINTR) continue;
            break;
        }

        int status = 0;
        pid_t waited = waitpid(env->strategy_pid, &status, WNOHANG);
        if (waited == env->strategy_pid) {
            close(env->strategy_output_fd);
            env->strategy_output_fd = -1;
            env->strategy_pid = -1;
            strategy_active_env = NULL;
            env->strategy_response[env->strategy_response_bytes] = '\0';

            int input_tokens;
            int output_tokens;
            if (parse_nonnegative_int(env->strategy_response, "llm_input_tokens", &input_tokens) &&
                    parse_nonnegative_int(env->strategy_response, "llm_output_tokens", &output_tokens)) {
                env->log.llm_input_tokens = (float)input_tokens;
                env->log.llm_output_tokens = (float)output_tokens;
                env->llm_cumulative_input_tokens += (unsigned int)input_tokens;
                env->llm_cumulative_output_tokens += (unsigned int)output_tokens;
                env->log.llm_cumulative_input_tokens =
                    (float)env->llm_cumulative_input_tokens;
                env->log.llm_cumulative_output_tokens =
                    (float)env->llm_cumulative_output_tokens;
            } else {
                input_tokens = -1;
            }

            if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && input_tokens >= 0) {
                float preferences[9];
                int horizon;
                if (parse_guidance(env->strategy_response, preferences, &horizon) &&
                        horizon <= INT_MAX - env->tick) {
                    memcpy(strategy_preferences, preferences, sizeof(preferences));
                    strategy_expire_tick = env->tick + horizon;
                    apply_strategy_preferences(env);
                    env->log.strategy_updates += 1.0f;
                    fprintf(stderr, "NMMO3 strategy preferences updated for %d ticks\n",
                        horizon);
                    result = 1;
                } else {
                    record_strategy_failure(env,
                        "response did not contain valid reward preferences");
                }
            } else {
                record_strategy_failure(env,
                    WIFEXITED(status) ? "strategy process returned an error" :
                        "strategy process terminated unexpectedly");
            }
        } else if (waited < 0 && errno != EINTR) {
            pid_t child = env->strategy_pid;
            kill(child, SIGTERM);
            waitpid(child, NULL, 0);
            close(env->strategy_output_fd);
            env->strategy_output_fd = -1;
            env->strategy_pid = -1;
            strategy_active_env = NULL;
            record_strategy_failure(env, "could not collect strategy process");
        }
        unlock_strategy();
        return result;
    }

    if (strategy_active_env == NULL && env->strategy_interval > 0 &&
            env->tick % env->strategy_interval == 0) {
        (void)launch_strategy(env, pid);
    }
    unlock_strategy();
    return result;
}

void nmmo3_strategy_close(MMO* env) {
    lock_strategy();
    if (env->strategy_pid > 0) {
        kill(env->strategy_pid, SIGTERM);
        waitpid(env->strategy_pid, NULL, 0);
        if (env->strategy_output_fd >= 0) close(env->strategy_output_fd);
        env->strategy_pid = -1;
        env->strategy_output_fd = -1;
    }
    if (strategy_active_env == env) strategy_active_env = NULL;
    unlock_strategy();
}
