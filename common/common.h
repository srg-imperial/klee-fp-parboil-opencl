
#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

bool getCmdLineParamInt(const char* keyword, int argc, char** argv, int *val);
bool getCmdLineParamBool(const char* keyword, int argc, char** argv);

typedef struct {
  struct timeval start;
  struct timeval stop;
} inf_timer;

void startTimer(inf_timer* t);
void stopTimer(inf_timer* t);
float elapsedTime(const inf_timer& t);

#endif

