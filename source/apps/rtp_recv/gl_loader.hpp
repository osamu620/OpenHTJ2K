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
  typedef struct __GLsync* GLsync;
  typedef unsigned long long GLuint64;

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
#    define GL_NUM_EXTENSIONS         0x821D
#    define GL_MAP_WRITE_BIT          0x0002
#    define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#    define GL_ALREADY_SIGNALED       0x911A
#    define GL_TIMEOUT_EXPIRED        0x911B
#    define GL_CONDITION_SATISFIED    0x911C
#    define GL_WAIT_FAILED            0x911D
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

// ARB_buffer_storage (GL 4.4) tokens are absent from the macOS headers
// (Apple GL stops at 4.1) and from our minimal Windows block above.
#ifndef GL_PIXEL_UNPACK_BUFFER
#  define GL_PIXEL_UNPACK_BUFFER     0x88EC
#endif
#ifndef GL_MAP_PERSISTENT_BIT
#  define GL_MAP_PERSISTENT_BIT      0x0040
#  define GL_MAP_COHERENT_BIT        0x0080
#endif

#ifndef APIENTRY
#  define APIENTRY
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
extern PFNGLUNIFORM1FPROC               Uniform1f;
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

// ── Optional zero-copy upload symbols ───────────────────────────────────
// GetStringi is GL 3.0, the sync trio is GL 3.2, MapBufferRange/UnmapBuffer
// are GL 3.0, and BufferStorage is GL 4.4 / ARB_buffer_storage.  Declared
// with inline function-pointer types (not PFNGL*PROC names) because the
// macOS headers don't ship typedefs past 4.1.  Resolved separately by
// load_zero_copy_functions(); any of these may legitimately stay null
// (the renderer then keeps its plane-vector upload path).
extern const GLubyte* (APIENTRY *GetStringi)(GLenum name, GLuint index);
extern void           (APIENTRY *BufferStorage)(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags);
extern void*          (APIENTRY *MapBufferRange)(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access);
extern GLboolean      (APIENTRY *UnmapBuffer)(GLenum target);
extern GLsync         (APIENTRY *FenceSync)(GLenum condition, GLbitfield flags);
extern GLenum         (APIENTRY *ClientWaitSync)(GLsync sync, GLbitfield flags, GLuint64 timeout);
extern void           (APIENTRY *DeleteSync)(GLsync sync);

// Resolve every pointer above via glfwGetProcAddress.  On success returns
// true; on the first failure logs the missing symbol to stderr and returns
// false (leaving the already-resolved pointers populated, since the caller
// is expected to treat this as fatal for the shader path).
bool load_functions();

// Resolve the optional zero-copy symbols.  Returns true only if ALL of
// them resolved; never fatal — callers treat false as "feature absent".
// Does not log: missing 4.4 entry points are expected on macOS.
bool load_zero_copy_functions();

}  // namespace open_htj2k::rtp_recv::gl
