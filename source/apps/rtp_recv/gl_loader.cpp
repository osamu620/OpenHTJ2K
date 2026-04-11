// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#include "gl_loader.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>

namespace open_htj2k::rtp_recv::gl {

PFNGLCREATESHADERPROC            CreateShader            = nullptr;
PFNGLSHADERSOURCEPROC            ShaderSource            = nullptr;
PFNGLCOMPILESHADERPROC           CompileShader           = nullptr;
PFNGLGETSHADERIVPROC             GetShaderiv             = nullptr;
PFNGLGETSHADERINFOLOGPROC        GetShaderInfoLog        = nullptr;
PFNGLDELETESHADERPROC            DeleteShader            = nullptr;

PFNGLCREATEPROGRAMPROC           CreateProgram           = nullptr;
PFNGLATTACHSHADERPROC            AttachShader            = nullptr;
PFNGLLINKPROGRAMPROC             LinkProgram             = nullptr;
PFNGLGETPROGRAMIVPROC            GetProgramiv            = nullptr;
PFNGLGETPROGRAMINFOLOGPROC       GetProgramInfoLog       = nullptr;
PFNGLUSEPROGRAMPROC              UseProgram              = nullptr;
PFNGLDELETEPROGRAMPROC           DeleteProgram           = nullptr;

PFNGLGETUNIFORMLOCATIONPROC      GetUniformLocation      = nullptr;
PFNGLUNIFORM1IPROC               Uniform1i               = nullptr;
PFNGLUNIFORM3FVPROC              Uniform3fv              = nullptr;
PFNGLUNIFORMMATRIX3FVPROC        UniformMatrix3fv        = nullptr;

PFNGLGENVERTEXARRAYSPROC         GenVertexArrays         = nullptr;
PFNGLBINDVERTEXARRAYPROC         BindVertexArray         = nullptr;
PFNGLDELETEVERTEXARRAYSPROC      DeleteVertexArrays      = nullptr;

PFNGLGENBUFFERSPROC              GenBuffers              = nullptr;
PFNGLBINDBUFFERPROC              BindBuffer              = nullptr;
PFNGLBUFFERDATAPROC              BufferData              = nullptr;
PFNGLDELETEBUFFERSPROC           DeleteBuffers           = nullptr;

PFNGLVERTEXATTRIBPOINTERPROC     VertexAttribPointer     = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray = nullptr;

PFNGLACTIVETEXTUREPROC           ActiveTexture           = nullptr;

namespace {
template <typename T>
bool resolve(T& out, const char* name) {
  out = reinterpret_cast<T>(glfwGetProcAddress(name));
  if (out == nullptr) {
    std::fprintf(stderr, "gl_loader: glfwGetProcAddress(\"%s\") returned null\n", name);
    return false;
  }
  return true;
}
}  // namespace

bool load_functions() {
#define LOAD(ptr, name)                    \
  do {                                     \
    if (!resolve(ptr, name)) return false; \
  } while (0)

  LOAD(CreateShader,            "glCreateShader");
  LOAD(ShaderSource,            "glShaderSource");
  LOAD(CompileShader,           "glCompileShader");
  LOAD(GetShaderiv,             "glGetShaderiv");
  LOAD(GetShaderInfoLog,        "glGetShaderInfoLog");
  LOAD(DeleteShader,            "glDeleteShader");

  LOAD(CreateProgram,           "glCreateProgram");
  LOAD(AttachShader,            "glAttachShader");
  LOAD(LinkProgram,             "glLinkProgram");
  LOAD(GetProgramiv,            "glGetProgramiv");
  LOAD(GetProgramInfoLog,       "glGetProgramInfoLog");
  LOAD(UseProgram,              "glUseProgram");
  LOAD(DeleteProgram,           "glDeleteProgram");

  LOAD(GetUniformLocation,      "glGetUniformLocation");
  LOAD(Uniform1i,               "glUniform1i");
  LOAD(Uniform3fv,              "glUniform3fv");
  LOAD(UniformMatrix3fv,        "glUniformMatrix3fv");

  LOAD(GenVertexArrays,         "glGenVertexArrays");
  LOAD(BindVertexArray,         "glBindVertexArray");
  LOAD(DeleteVertexArrays,      "glDeleteVertexArrays");

  LOAD(GenBuffers,              "glGenBuffers");
  LOAD(BindBuffer,              "glBindBuffer");
  LOAD(BufferData,              "glBufferData");
  LOAD(DeleteBuffers,           "glDeleteBuffers");

  LOAD(VertexAttribPointer,     "glVertexAttribPointer");
  LOAD(EnableVertexAttribArray, "glEnableVertexAttribArray");

  LOAD(ActiveTexture,           "glActiveTexture");
#undef LOAD
  return true;
}

}  // namespace open_htj2k::rtp_recv::gl
