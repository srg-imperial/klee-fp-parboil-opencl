
#include "ocl-wrapper.h"

#include <stdio.h>
#include <string.h>

#include <cstdlib>

OclWrapper::OclWrapper ()
{
}


OclWrapper::~OclWrapper ()
{
}


bool OclWrapper::init (const char *platform, int device)
{
  // initialize OpenCL context
  cl_int err;

  // query platform IDs
  unsigned int num_ids;
  err = clGetPlatformIDs (0, NULL, &num_ids);
  if (!checkErr(err, "failed to collect number of available platforms"))
    return false;
  cl_platform_id *platform_ids = (cl_platform_id*) malloc (num_ids * sizeof(cl_platform_id));
  err = clGetPlatformIDs (num_ids, platform_ids, NULL);
  if (!checkErr(err, "failed to collect platform IDs"))
    return false;

  // select specified platform
  unsigned int platform_idx = 0;
  char platform_name[256];
  for (platform_idx = 0; platform_idx < num_ids; platform_idx++) {
    err = clGetPlatformInfo (platform_ids[platform_idx], CL_PLATFORM_NAME, 256*sizeof(char), platform_name, NULL);
    if (!checkErr (err, "failed to read platform name"))
      return false;
    if (strcmp (platform, platform_name) == 0) { // platform found
      fprintf (stderr, "platform name: %s\n", platform_name);
      break;
    }
  }
  if (platform_idx == num_ids) {
    fprintf (stderr, "invalid platform: %s\n", platform);
    return false;
  }

  // query devices on selected platform
  err = clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_ALL, 0, NULL, &num_ids);
  if (!checkErr(err, "failed to collect number of available devices"))
    return false;
  cl_device_id *device_ids = (cl_device_id*) malloc (num_ids * sizeof (cl_device_id));
  err = clGetDeviceIDs(platform_ids[platform_idx], CL_DEVICE_TYPE_ALL, num_ids, device_ids, NULL);
  if (!checkErr(err, "failed to collect device IDs"))
    return false;

  // select specified device
  if (device >= num_ids)
  {
    fprintf (stderr, "invalid device ID: %d (only %d devices found on platform %s)", device, num_ids, platform);
    return false;
  }

  this->dev_id = device_ids[device];

  // create context and command queue
  fprintf (stderr, "create context..\n");

  this->ctxt = clCreateContext(0, 1, &this->dev_id, NULL, NULL, &err);
  if (!checkErr(err, "failed to create context..."))
    return false;

  fprintf (stderr, "create command queue..\n");

  this->cmd_queue = clCreateCommandQueue (ctxt, this->dev_id, CL_QUEUE_PROFILING_ENABLE, &err);
  if (!checkErr(err, "failed to create command queue..."))
    return false;

  free (platform_ids);
  free (device_ids);

  return true;

}


// load and build kernel
bool OclWrapper::buildProgram(const char* filename, cl_program* prog)
{
  size_t file_length;
  char *source = loadFile(filename, &file_length);
  cl_int err;

  if (!source)
    return false;

  *prog = clCreateProgramWithSource (this->ctxt, 1, (const char**)&source, &file_length, &err);
  if (!checkErr(err, "cannot load program")) {
    free(source);
    // read build log
    char log[1024];
    err = clGetProgramBuildInfo (*prog, this->dev_id, CL_PROGRAM_BUILD_LOG, 1024 * sizeof(char), &log, NULL);
    fprintf (stderr, "%s\n", log);
    return false;
  }

  err = clBuildProgram(*prog, 0, NULL, "-cl-mad-enable", NULL, NULL);
  if (!checkErr(err, "cannot build program")) {
    free(source);
    // read build log
    char log[1024];
    err = clGetProgramBuildInfo (*prog, this->dev_id, CL_PROGRAM_BUILD_LOG, 1024 * sizeof(char), &log, NULL);
    fprintf (stderr, "%s\n", log);
    return false;
  }

  free(source);
  return true;

}

// read entire file and return contents and file length
char* OclWrapper::loadFile(const char* filename, size_t* fileLength)
{
  // open file
  FILE* file = fopen(filename, "r");

  if (!file) {
    fprintf (stderr, "cannot open file: %s\n", filename);
    return NULL;
  }

  // get file length
  fseek (file, 0, SEEK_END);
  *fileLength = ftell(file);
  fseek (file, 0, SEEK_SET);

  // allocate memory for file contents
  char* content = (char*) malloc (*fileLength * sizeof(char));

  // read entire file in one go
  size_t bytes_read = fread(content, sizeof(char), *fileLength,file);

  // close file
  fclose(file);

  return content;

}



bool OclWrapper::checkErr(cl_int err, const char* err_msg)
{
  if (err == CL_SUCCESS)
    return true;

  // some error has occurred
  if (err_msg)
    fprintf (stderr, "%s\n\terrno: %d\n", err_msg, err);
  else
    fprintf (stderr, "error!");

  // decipher error code
  switch (err)
  {
  case CL_SUCCESS: fprintf (stderr, "\tCL_SUCCESS\t\n"); break;
  case CL_DEVICE_NOT_FOUND: fprintf (stderr, "\tCL_DEVICE_NOT_FOUND\t\n"); break;
  case CL_DEVICE_NOT_AVAILABLE: fprintf (stderr, "\tCL_DEVICE_NOT_AVAILABLE\t\n"); break;
  case CL_COMPILER_NOT_AVAILABLE: fprintf (stderr, "\tCL_COMPILER_NOT_AVAILABLE\t\n"); break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE: fprintf (stderr, "\tCL_MEM_OBJECT_ALLOCATION_FAILURE\t\n"); break;
  case CL_OUT_OF_RESOURCES: fprintf (stderr, "\tCL_OUT_OF_RESOURCES\t\n"); break;
  case CL_OUT_OF_HOST_MEMORY: fprintf (stderr, "\tCL_OUT_OF_HOST_MEMORY\t\n"); break;
  case CL_PROFILING_INFO_NOT_AVAILABLE: fprintf (stderr, "\tCL_PROFILING_INFO_NOT_AVAILABLE\t\n"); break;
  case CL_MEM_COPY_OVERLAP: fprintf (stderr, "\tCL_MEM_COPY_OVERLAP\t\n"); break;
  case CL_IMAGE_FORMAT_MISMATCH: fprintf (stderr, "\tCL_IMAGE_FORMAT_MISMATCH\t\n"); break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED: fprintf (stderr, "\tCL_IMAGE_FORMAT_NOT_SUPPORTED\t\n"); break;
  case CL_BUILD_PROGRAM_FAILURE: fprintf (stderr, "\tCL_BUILD_PROGRAM_FAILURE\t\n"); break;
  case CL_MAP_FAILURE: fprintf (stderr, "\tCL_MAP_FAILURE\t\n"); break;
  case CL_INVALID_VALUE: fprintf (stderr, "\tCL_INVALID_VALUE\t\n"); break;
  case CL_INVALID_DEVICE_TYPE: fprintf (stderr, "\tCL_INVALID_DEVICE_TYPE\t\n"); break;
  case CL_INVALID_PLATFORM: fprintf (stderr, "\tCL_INVALID_PLATFORM\t\n"); break;
  case CL_INVALID_DEVICE: fprintf (stderr, "\tCL_INVALID_DEVICE\t\n"); break;
  case CL_INVALID_CONTEXT: fprintf (stderr, "\tCL_INVALID_CONTEXT\t\n"); break;
  case CL_INVALID_QUEUE_PROPERTIES: fprintf (stderr, "\tCL_INVALID_QUEUE_PROPERTIES\t\n"); break;
  case CL_INVALID_COMMAND_QUEUE: fprintf (stderr, "\tCL_INVALID_COMMAND_QUEUE\t\n"); break;
  case CL_INVALID_HOST_PTR: fprintf (stderr, "\tCL_INVALID_HOST_PTR\t\n"); break;
  case CL_INVALID_MEM_OBJECT: fprintf (stderr, "\tCL_INVALID_MEM_OBJECT\t\n"); break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: fprintf (stderr, "\tCL_INVALID_IMAGE_FORMAT_DESCRIPTOR\t\n"); break;
  case CL_INVALID_IMAGE_SIZE: fprintf (stderr, "\tCL_INVALID_IMAGE_SIZE\t\n"); break;
  case CL_INVALID_SAMPLER: fprintf (stderr, "\tCL_INVALID_SAMPLER\t\n"); break;
  case CL_INVALID_BINARY: fprintf (stderr, "\tCL_INVALID_BINARY\t\n"); break;
  case CL_INVALID_BUILD_OPTIONS: fprintf (stderr, "\tCL_INVALID_BUILD_OPTIONS\t\n"); break;
  case CL_INVALID_PROGRAM: fprintf (stderr, "\tCL_INVALID_PROGRAM\t\n"); break;
  case CL_INVALID_PROGRAM_EXECUTABLE: fprintf (stderr, "\tCL_INVALID_PROGRAM_EXECUTABLE\t\n"); break;
  case CL_INVALID_KERNEL_NAME: fprintf (stderr, "\tCL_INVALID_KERNEL_NAME\t\n"); break;
  case CL_INVALID_KERNEL_DEFINITION: fprintf (stderr, "\tCL_INVALID_KERNEL_DEFINITION\t\n"); break;
  case CL_INVALID_KERNEL: fprintf (stderr, "\tCL_INVALID_KERNEL\t\n"); break;
  case CL_INVALID_ARG_INDEX: fprintf (stderr, "\tCL_INVALID_ARG_INDEX\t\n"); break;
  case CL_INVALID_ARG_VALUE: fprintf (stderr, "\tCL_INVALID_ARG_VALUE\t\n"); break;
  case CL_INVALID_ARG_SIZE: fprintf (stderr, "\tCL_INVALID_ARG_SIZE\t\n"); break;
  case CL_INVALID_KERNEL_ARGS: fprintf (stderr, "\tCL_INVALID_KERNEL_ARGS\t\n"); break;
  case CL_INVALID_WORK_DIMENSION: fprintf (stderr, "\tCL_INVALID_WORK_DIMENSION\t\n"); break;
  case CL_INVALID_WORK_GROUP_SIZE: fprintf (stderr, "\tCL_INVALID_WORK_GROUP_SIZE\t\n"); break;
  case CL_INVALID_WORK_ITEM_SIZE: fprintf (stderr, "\tCL_INVALID_WORK_ITEM_SIZE\t\n"); break;
  case CL_INVALID_GLOBAL_OFFSET: fprintf (stderr, "\tCL_INVALID_GLOBAL_OFFSET\t\n"); break;
  case CL_INVALID_EVENT_WAIT_LIST: fprintf (stderr, "\tCL_INVALID_EVENT_WAIT_LIST\t\n"); break;
  case CL_INVALID_EVENT: fprintf (stderr, "\tCL_INVALID_EVENT\t\n"); break;
  case CL_INVALID_OPERATION: fprintf (stderr, "\tCL_INVALID_OPERATION\t\n"); break;
  case CL_INVALID_GL_OBJECT: fprintf (stderr, "\tCL_INVALID_GL_OBJECT\t\n"); break;
  case CL_INVALID_BUFFER_SIZE: fprintf (stderr, "\tCL_INVALID_BUFFER_SIZE\t\n"); break;
  case CL_INVALID_MIP_LEVEL: fprintf (stderr, "\tCL_INVALID_MIP_LEVEL\t\n"); break;
  default: fprintf (stderr, "\tunknown error...\n");
  }

  return false;
}




