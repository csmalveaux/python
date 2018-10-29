
import math
from glumpy import app, gloo, gl

vertex = """
  uniform float scale;
  uniform float theta;
  attribute vec2 position;
  attribute vec4 color;
  varying vec4 v_color;
  void main() {
      float ct = cos(theta);
      float st = sin(theta);
      float x = 0.75 * (position.x*ct - position.y*st);
      float y = 0.75 * (position.x*st + position.y*ct);
      gl_Position = vec4(x, y, 0.0, 1.0);
      v_color = color;
  }
 """

fragment = """
  varying vec4 v_color;
  void main() {
      gl_FragColor = v_color;
  }
"""

# Create a window with a valid GL context
window = app.Window()

quad = gloo.Program(vertex, fragment, count=4)
quad['position'] = (-1, +1), (+1, +1), (-1, -1), (+1, -1)
quad['color'] = (1, 1, 0, 1), (1, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1)
quad['scale'] = 1.0
quad['theta'] = 0.0

time = 0.0
theta = 0.0

@window.event
def on_draw(dt):
    global time
    time += 1.0 * math.pi/180.0
    window.clear()
    quad["scale"] = math.cos(time)
    quad["theta"] += dt
    quad.draw(gl.GL_TRIANGLE_STRIP)


app.run()
