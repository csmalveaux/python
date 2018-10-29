import math
from glumpy import app, gloo, gl, glm
from glumpy.geometry import colorcube
import numpy as np

vertex = """
    uniform mat4   model;
    uniform mat4   view;
    uniform mat4   projection;
    attribute vec4 color;
    attribute vec3 position;
    varying vec4 v_color;
    void main()
    {
        v_color = color;
        gl_Position = projection * view * model * vec4(position,1.0);
    }
"""

fragment = """
  varying vec4 v_color;
  void main() {
      gl_FragColor = v_color;
  }
"""

window = app.Window(width=512, height=512, color=(1, 1, 1, 1))


@window.event
def on_draw(dt):
    global phi, theta, shift
    window.clear()

    # Filled cube
    cube.draw(gl.GL_TRIANGLES, indices)

    # Make cube rotate
    #theta += 1.0  # degrees
    #phi += 1.0  # degrees
    shift += 0.05
    model = glm.translation(0, 0, shift)
    glm.rotate(model, theta, 0, 0, 1)
    glm.rotate(model, phi, 0, 1, 0)
    cube['model'] = model


@window.event
def on_resize(width, height):
    cube['projection'] = glm.perspective(
        45.0, width / float(height), 2.0, 100.0)


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)


vertices = np.zeros(8, [("position", np.float32, 3), ("color", np.float32, 4)])
vertices['position'] = [[1, 1, 1], [-1, 1, 1], [-1, -1, 1],
                        [1, -1, 1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]]
vertices['color'] = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [
    0, 1, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]

indices = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1,
                    1, 6, 7, 1, 7, 2, 7, 4, 3, 7, 3, 2, 4, 7, 6, 4, 6, 5], dtype=np.uint32)

vertices = vertices.view(gloo.VertexBuffer)
indices = indices.view(gloo.IndexBuffer)

cube = gloo.Program(vertex, fragment)
cube.bind(vertices)
cube['model'] = np.eye(4, dtype=np.float32)
cube['view'] = glm.translation(0, 0, -50)
phi, theta = 40, 30
shift = 10

app.run(framerate=60, framecount=360)
