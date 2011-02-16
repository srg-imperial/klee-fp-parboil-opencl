
#include "common.h"

#include <string.h>


// search command line parameters for keyword
// check if integer value is following
bool getCmdLineParamInt(const char* keyword, int argc, char** argv, int *val)
{
  int v;

  for (int i=1; i<argc-1; i++)
  {
    if (strcmp(argv[i], keyword) == 0)
    {
      v = atoi(argv[i+1]);
      if (v > 0)
      {
        if (val) *val=v;
        return true;
      }
      else
      {
        printf ("invalid input for %s: %s (must be positive integer)\n", argv[i], argv[i+1]);
        return false;
      }
    }
  }

  // keyword not found
  return false;
}

// search command line parameters for keyword
bool getCmdLineParamBool(const char* keyword, int argc, char** argv)
{
  for (int i=1; i<argc; i++)
  {
    if (strcmp(argv[i], keyword) == 0)
      return true;
  }

  // keyword not found
  return false;
}

void startTimer(inf_timer* t)
{
  gettimeofday(&t->start, NULL);
}

void stopTimer(inf_timer* t)
{
  gettimeofday(&t->stop, NULL);
}

float elapsedTime(const inf_timer& t)
{
  return (t.stop.tv_sec - t.start.tv_sec) * 1000.0 + 0.001 * (t.stop.tv_usec - t.start.tv_usec);
}


