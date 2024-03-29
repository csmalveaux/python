import sys
import ctypes
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import numpy as np

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    glut.glutSwapBuffers()


def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()

def Square():
	data = np.zeros(4, [("position", np.float32, 2), ("color", np.float32, 4)])
	data['position'] = [(-1, +1), (+1, +1), (-1, -1), (+1, -1)]
	data['color'] = [(0, 1, 0, 1), (1, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1)]
	program = gl.glCreateProgram()
	vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
	fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

	vertex_code = ""
	with open('vertexcode1.vs', 'r') as f:
		vertex_code = f.read()

	fragment_code = ""
	with open('fragmentcode1.fs', 'r') as f:
		fragment_code = f.read()

	gl.glShaderSource(vertex, vertex_code)
	gl.glShaderSource(fragment, fragment_code)

	gl.glCompileShader(vertex)
	if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
	    error = gl.glGetShaderInfoLog(vertex).decode()
	    print(error)
	    raise RuntimeError("Shader compilation error")

	gl.glCompileShader(fragment)
	if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
	    error = gl.glGetShaderInfoLog(fragment).decode()
	    print(error)
	    raise RuntimeError("Fragment compilation error")

	gl.glAttachShader(program, vertex)
	gl.glAttachShader(program, fragment)

	gl.glLinkProgram(program)
	if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
	    print(gl.glGetProgramInfoLog(program))
	    raise RuntimeError('Linking error')

	gl.glDetachShader(program, vertex)
	gl.glDetachShader(program, fragment)

	gl.glUseProgram(program)

	buffer_ = gl.glGenBuffers(1)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_)
	gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

	stride = data.strides[0]
	offset = ctypes.c_void_p(0)
	loc = gl.glGetAttribLocation(program, "position")
	gl.glEnableVertexAttribArray(loc)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_)
	gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, offset)

	offset = ctypes.c_void_p(data.dtype["position"].itemsize)
	loc = gl.glGetAttribLocation(program, "color")
	gl.glEnableVertexAttribArray(loc)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_)
	gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)


def main():
	glut.glutInit()
	glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
	glut.glutCreateWindow('Hello world!')
	glut.glutReshapeWindow(512, 512)
	glut.glutReshapeFunc(reshape)
	glut.glutDisplayFunc(display)
	glut.glutKeyboardFunc(keyboard)
	
	Square()
	
	glut.glutMainLoop()


main()
