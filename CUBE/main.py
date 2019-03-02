#!/usr/bin/env python 3
# import os

import loaddata
loaddata.init()
from loaddata import trapRooms

import glfw
import OpenGL.GL as GL
# import OpenGL.GLUT as GLUT
# import OpenGL.GLU as GLU
# import numpy as np


class Viewer:
    def __init__(self, width=640, height=480):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        glfw.make_context_current(self.win)

        glfw.set_key_callback(self.win, self.on_key)

        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        GL.glClearColor(0.1, 0.1, 0.1, 0.1)

        self.drawables = []

    def run(self):
        while not glfw.window_should_close(self.win):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            for drawable in self.drawables:
                drawable.draw(None, None, None, self.color_shader)

            glfw.swap_buffers(self.win)
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):

        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)


def main():
    viewer = Viewer()

    viewer.run()

    print(loaddata.trapRooms)
if __name__ == '__main__':
    glfw.init()
    main()
    glfw.terminate()
