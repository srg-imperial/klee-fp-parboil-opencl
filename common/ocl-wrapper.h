
#ifndef OCL_WRAPPER_H
#define OCL_WRAPPER_H


#include <CL/cl.h>

class OclWrapper
{
public:
  OclWrapper();
  ~OclWrapper();

  bool init (const char *platform, int device_id);
  bool buildProgram(const char* filename, cl_program* prog);

  static bool checkErr (cl_int err, const char* err_msg);

  cl_context getContext() { return ctxt; }
  cl_command_queue getCmdQueue() { return cmd_queue; }

private:
  cl_context ctxt;
  cl_command_queue cmd_queue;
  cl_device_id dev_id;

  char* loadFile(const char* filename, size_t* fileLength);

};

#endif

