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
#elif defined(_WIN32)
  // Windows SDK ships GL/gl.h (GL 1.1 only) but not GL/glext.h.
  // Including GL/gl.h directly after winsock2.h (which pulls in windows.h)
  // also causes C2086 redefinition errors.  GLFW handles both issues:
  // it includes GL/gl.h in the correct order and provides the base types.
  // We declare the ~25 GL 2.0+ function-pointer typedefs we need below,
  // avoiding a dependency on a third-party glext.h.
#  include <GLFW/glfw3.h>

  // GL types not in Windows GL 1.1 headers:
  typedef char         GLchar;
  typedef ptrdiff_t    GLsizeiptr;
  typedef ptrdiff_t    GLintptr;

  // GL constants not in Windows GL 1.1 headers:
#  ifndef GL_FRAGMENT_SHADER
#    define GL_FRAGMENT_SHADER        0x8B30
#    define GL_VERTEX_SHADER          0x8B31
#    define GL_COMPILE_STATUS         0x8B81
#    define GL_LINK_STATUS            0x8B82
#    define GL_INFO_LOG_LENGTH        0x8B84
#    define GL_ARRAY_BUFFER           0x8892
#    define GL_STATIC_DRAW            0x88E4
#    define GL_TEXTURE0               0x84C0
#    define GL_TEXTURE1               0x84C1
#    define GL_TEXTURE2               0x84C2
#    define GL_R8                     0x8229
#    define GL_R16                    0x822A
#    define GL_RED                    0x1903
#    define GL_CLAMP_TO_EDGE          0x812F
#  endif

  // GL 2.0+ function-pointer typedefs:
  typedef GLuint  (APIENTRY *PFNGLCREATESHADERPROC)(GLenum type);
  typedef void    (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const *string, const GLint *length);
  typedef void    (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint shader);
  typedef void    (APIENTRY *PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
  typedef void    (APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
  typedef void    (APIENTRY *PFNGLDELETESHADERPROC)(GLuint shader);
  typedef GLuint  (APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
  typedef void    (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
  typedef void    (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint program);
  typedef void    (APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
  typedef void    (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
  typedef void    (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint program);
  typedef void    (APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint program);
  typedef GLint   (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
  typedef void    (APIENTRY *PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
  typedef void    (APIENTRY *PFNGLUNIFORM1FPROC)(GLint location, GLfloat v0);
  typedef void    (APIENTRY *PFNGLUNIFORM3FVPROC)(GLint location, GLsizei count, const GLfloat *value);
  typedef void    (APIENTRY *PFNGLUNIFORMMATRIX3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
  typedef void    (APIENTRY *PFNGLGENVERTEXARRAYSPROC)(GLsizei n, GLuint *arrays);
  typedef void    (APIENTRY *PFNGLBINDVERTEXARRAYPROC)(GLuint array);
  typedef void    (APIENTRY *PFNGLDELETEVERTEXARRAYSPROC)(GLsizei n, const GLuint *arrays);
  typedef void    (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
  typedef void    (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
  typedef void    (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
  typedef void    (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
  typedef void    (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
  typedef void    (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
  typedef void    (APIENTRY *PFNGLACTIVETEXTUREPROC)(GLenum texture);
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
