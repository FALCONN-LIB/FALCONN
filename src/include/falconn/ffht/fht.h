#ifndef _FHT_H_
#define _FHT_H_

#ifdef __cplusplus
extern "C" {
#endif

int fht_float(float *buf, int log_n);
int fht_double(double *buf, int log_n);

#ifdef __cplusplus
}
#endif

#endif
