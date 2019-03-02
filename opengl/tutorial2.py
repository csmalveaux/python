import sys
import ctypes
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import numpy as np
import math
from glumpy import app

window = app.Window(width=512, height=512, color=(1, 1, 1, 1))

def frustum(left, right, bottom, top, znear, zfar):
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    assert(znear != zfar)
    h = math.tan(fovy / 360.0 * math.pi) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(M, x, y=None, z=None):
    y = x if y is None else y
    z = x if z is None else z
    T = np.array([[1.0, 0.0, 0.0, x],
                  [0.0, 1.0, 0.0, y],
                  [0.0, 0.0, 1.0, z],
                  [0.0, 0.0, 0.0, 1.0]], dtype=M.dtype).T
    M[...] = np.dot(M, T)
    return M


def translation(x, y=None, z=None):
    M = np.eye(4, dtype=np.float32)
    return translate(M,x,y,z)

def display():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LESS)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
	#gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 9)
    gl.glDrawElements(gl.GL_TRIANGLES, len(index), gl.GL_UNSIGNED_INT, None)
    glut.glutSwapBuffers()


def reshape(width, height):
    gl.glViewport(0, 0, width, height)


def keyboard(key, x, y):
    if key == b'\x1b':
        sys.exit()

def Cube():
	global index
	data = np.zeros(8, [("position", np.float32, 3), ("color", np.float32, 4)])
	data['position'] = [[ 1, 1, 1], [-1, 1, 1], [-1,-1, 1], [ 1,-1, 1],
                   [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], [-1,-1,-1]]
	data['color'] = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [
    				0, 1, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]
	index = np.array( [0,1,2, 
                  0,2,3,  
                  0,3,4, 
                  0,4,5,  
                  0,5,6, 
                  0,6,1,
                  1,6,7, 
                  1,7,2,  
                  7,4,3, 
                  7,3,2,  
                  4,7,6, 
                  4,6,5], dtype=np.uint32)

	model = np.eye(4, dtype=np.float32)
	view = translation(0, 0, -25)
	projection = perspective(45.0, width / float(height), 2.0, 100.0)

	program = gl.glCreateProgram()
	vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
	fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

	vertex_code = ""
	with open('vertexcode2.vs', 'r') as f:
		vertex_code = f.read()

	fragment_code = ""
	with open('fragmentcode2.fs', 'r') as f:
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

	buffer_index = gl.glGenBuffers(1)
	gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer_index)
	gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index.nbytes, index, gl.GL_STATIC_DRAW)

	stride = data.strides[0]
	offset = ctypes.c_void_p(0)
	loc = gl.glGetAttribLocation(program, "position")
	gl.glEnableVertexAttribArray(loc)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_)
	gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, offset)

	offset = ctypes.c_void_p(data.dtype["position"].itemsize)
	loc = gl.glGetAttribLocation(program, "color")
	gl.glEnableVertexAttribArray(loc)
	gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer_)
	gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, offset)

	gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, buffer_index)

	gl.glEnable(gl.GL_DEPTH_TEST)
	gl.glDepthFunc(gl.GL_LESS)

	loc = gl.glGetUniformLocation(program, "model")
	gl.glUniformMatrix4fv(loc, 1, False, model)

	loc = gl.glGetUniformLocation(program, "view")
	gl.glUniformMatrix4fv(loc, 1, False, view)

	loc = gl.glGetUniformLocation(program, "projection")
	gl.glUniformMatrix4fv(loc, 1, False, projection)


def main():
	global width 
	global height
	width=512
	height=512
	glut.glutInit()
	glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA)
	glut.glutCreateWindow('Hello world!')
	glut.glutReshapeWindow(width, height)
	glut.glutReshapeFunc(reshape)
	glut.glutDisplayFunc(display)
	glut.glutKeyboardFunc(keyboard)
	
	Cube()
	
	glut.glutMainLoop()


main()
