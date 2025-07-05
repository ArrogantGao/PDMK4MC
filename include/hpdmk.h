#ifndef HPDMK_H
#define HPDMK_H

typedef enum : int {
    HPDMK_LOG_TRACE = 0,
    HPDMK_LOG_DEBUG = 1,
    HPDMK_LOG_INFO = 2,
    HPDMK_LOG_WARN = 3,
    HPDMK_LOG_ERR = 4,
    HPDMK_LOG_CRITICAL = 5,
    HPDMK_LOG_OFF = 6,
} hpdmk_log_level;

typedef struct hpdmk_params {
    int n_dim = 3;
    int log_level = 6;               // 0: trace, 1: debug, 2: info, 3: warn, 4: err, 5: critical, 6: off
} hpdmk_params;

#ifdef __cplusplus
extern "C" {
#endif

void hpdmk_init(hpdmk_params params);
void hpdmk_finalize();

#ifdef __cplusplus
}
#endif

#endif
