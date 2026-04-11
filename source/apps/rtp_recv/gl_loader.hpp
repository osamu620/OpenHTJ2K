// Copyright (c) 2026, Osamu Watanabe
// All rights reserved.
//
// Licensed under the same BSD 3-Clause terms as the rest of OpenHTJ2K.

#pragma once

// Minimal GL function-pointer loader for the rtp_recv shader path.
//
// Everything GL >=2.0 has to be resolved via glfwGetProcAddress on Linux
// because <GL/gl.h> only exports through ~GL 1.3.  Rather than depend on
// glad or GLEW for the ~20 symbols we actually use, this header declares
// a small set of function pointers in open_htj2k::rtp_recv::gl and
// gl_loader.cpp resolves them once in load_functions().
//
// Call load_functions() after glfwMakeContextCurrent().  Returns false if
// any required symbol is missing (the caller should log and fall back to
// the CPU RGB path, which is what GL-incompatible environments will use).

#if defined(__APPLE__)
#  define GL_SILENCE_DEPRECATION
#  include <OpenGL/gl3.h>
#  include <OpenGL/gl3ext.h>
#else
#  include <GL/gl.h>
#  include <GL/glext.h>
#endif

namespace open_htj2k::rtp_recv::gl {

extern PFNGLCREATESHADERPROC            CreateShader;
extern PFNGLSHADERSOURCEPROC            ShaderSource;
extern PFNGLCOMPILESHADERPROC           CompileShader;
extern PFNGLGETSHADERIVPROC             GetShaderiv;
extern PFNGLGETSHADERINFOLOGPROC        GetShaderInfoLog;
extern PFNGLDELETESHADERPROC            DeleteShader;

extern PFNGLCREATEPROGRAMPROC           CreateProgram;
extern PFNGLATTACHSHADERPROC            AttachShader;
extern PFNGLLINKPROGRAMPROC             LinkProgram;
extern PFNGLGETPROGRAMIVPROC            GetProgramiv;
extern PFNGLGETPROGRAMINFOLOGPROC       GetProgramInfoLog;
extern PFNGLUSEPROGRAMPROC              UseProgram;
extern PFNGLDELETEPROGRAMPROC           DeleteProgram;

extern PFNGLGETUNIFORMLOCATIONPROC      GetUniformLocation;
extern PFNGLUNIFORM1IPROC               Uniform1i;
extern PFNGLUNIFORM3FVPROC              Uniform3fv;
extern PFNGLUNIFORMMATRIX3FVPROC        UniformMatrix3fv;

extern PFNGLGENVERTEXARRAYSPROC         GenVertexArrays;
extern PFNGLBINDVERTEXARRAYPROC         BindVertexArray;
extern PFNGLDELETEVERTEXARRAYSPROC      DeleteVertexArrays;

extern PFNGLGENBUFFERSPROC              GenBuffers;
extern PFNGLBINDBUFFERPROC              BindBuffer;
extern PFNGLBUFFERDATAPROC              BufferData;
extern PFNGLDELETEBUFFERSPROC           DeleteBuffers;

extern PFNGLVERTEXATTRIBPOINTERPROC     VertexAttribPointer;
extern PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray;

extern PFNGLACTIVETEXTUREPROC           ActiveTexture;

// Resolve every pointer above via glfwGetProcAddress.  On success returns
// true; on the first failure logs the missing symbol to stderr and returns
// false (leaving the already-resolved pointers populated, since the caller
// is expected to treat this as fatal for the shader path).
bool load_functions();

}  // namespace open_htj2k::rtp_recv::gl
