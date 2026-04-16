package main

import (
	"log"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

const (
	winWidth  = 1280
	winHeight = 720

	nParticles = 2000
)

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
}

var (
	window *glfw.Window

	winW = float32(winWidth)
	winH = float32(winHeight)

	// camera
	camPos   = mgl32.Vec3{0, 0, 800}
	camYaw   float64
	camPitch float64

	keys      = map[glfw.Key]bool{}
	mouseDown bool
	lastX     float64
	lastY     float64

	// particles
	particles []Particle
	vao       uint32
	vbo       uint32
	prog      uint32

	// screen
	screenProg uint32
	quadVAO    uint32
	quadVBO    uint32

	// FBO
	sceneFBO uint32
	sceneTex uint32
	depthRB  uint32

	// DRS
	dprScale      float32 = 1.0
	targetFPS             = 60.0
	lastFrameTime         = time.Now()
)

func init() { runtime.LockOSThread() }

func main() {

	if err := glfw.Init(); err != nil {
		log.Fatal(err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)

	window, _ = glfw.CreateWindow(winWidth, winHeight, "Particles Engine", nil, nil)
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(err)
	}

	gl.Enable(gl.PROGRAM_POINT_SIZE)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE)
	gl.Enable(gl.DEPTH_TEST)

	setupParticles()
	setupQuad()
	setupFBO(winWidth, winHeight)

	prog = mustProgram(particleVS, particleFS)
	screenProg = mustProgram(screenVS, screenFS)
	window, _ = glfw.CreateWindow(winWidth, winHeight, "Particles Engine", nil, nil)
	window.MakeContextCurrent()

	// input
	window.SetKeyCallback(keyCallback)
	window.SetCursorPosCallback(mouseMove)
	window.SetMouseButtonCallback(mouseButton)
	window.SetCursorPos(float64(winWidth)/2, float64(winHeight)/2)
	window.SetFramebufferSizeCallback(func(w *glfw.Window, width, height int) {
		winW = float32(width)
		winH = float32(height)
		setupFBO(width, height)
	})

	for !window.ShouldClose() {

		now := time.Now()
		dt := now.Sub(lastFrameTime).Seconds()
		lastFrameTime = now

		updateCamera(dt)
		updateParticles(dt)
		updateDRS(dt)

		render()

		window.SwapBuffers()
		glfw.PollEvents()
	}
}

//
// ---------------- PARTICLES ----------------
//

func setupParticles() {
	particles = make([]Particle, nParticles)

	for i := range particles {
		particles[i].Pos = mgl32.Vec3{
			float32(rand.Float64()*400 - 200),
			float32(rand.Float64()*400 - 200),
			float32(rand.Float64()*400 - 200),
		}
	}

	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)

	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, nParticles*3*4, nil, gl.DYNAMIC_DRAW)

	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 0, nil)
}

func updateParticles(dt float64) {
	for i := range particles {
		p := &particles[i]

		p.Vel = p.Vel.Add(mgl32.Vec3{
			float32(rand.Float64()-0.5) * 2,
			float32(rand.Float64()-0.5) * 2,
			float32(rand.Float64()-0.5) * 2,
		}.Mul(float32(dt * 50)))

		p.Pos = p.Pos.Add(p.Vel.Mul(float32(dt)))
		p.Vel = p.Vel.Mul(0.98)
	}
}

//
// ---------------- CAMERA ----------------
//

func updateCamera(dt float64) {

	speed := float32(300 * dt)
	forward := getForward()
	right := forward.Cross(mgl32.Vec3{0, 1, 0}).Normalize()

	if keys[glfw.KeyW] {
		camPos = camPos.Add(forward.Mul(speed))
	}
	if keys[glfw.KeyS] {
		camPos = camPos.Sub(forward.Mul(speed))
	}
	if keys[glfw.KeyA] {
		camPos = camPos.Sub(right.Mul(speed))
	}
	if keys[glfw.KeyD] {
		camPos = camPos.Add(right.Mul(speed))
	}
}

func getForward() mgl32.Vec3 {
	return mgl32.Vec3{
		float32(math.Cos(camYaw) * math.Cos(camPitch)),
		float32(math.Sin(camPitch)),
		float32(math.Sin(camYaw) * math.Cos(camPitch)),
	}
}

func getView() mgl32.Mat4 {
	f := getForward()
	return mgl32.LookAtV(camPos, camPos.Add(f), mgl32.Vec3{0, 1, 0})
}

//
// ---------------- INPUT ----------------
//

func keyCallback(w *glfw.Window, key glfw.Key, scancode int, action glfw.Action, mods glfw.ModifierKey) {
	if action == glfw.Press {
		keys[key] = true
	}
	if action == glfw.Release {
		keys[key] = false
	}
}

func mouseMove(w *glfw.Window, x, y float64) {
	if mouseDown {
		camYaw -= (x - lastX) * 0.002
		camPitch -= (y - lastY) * 0.002
	}
	lastX, lastY = x, y
}

func mouseButton(w *glfw.Window, b glfw.MouseButton, a glfw.Action, m glfw.ModifierKey) {
	mouseDown = (b == glfw.MouseButtonLeft && a == glfw.Press)
}

//
// ---------------- RENDER ----------------
//

func render() {

	w := int(float32(winW) * dprScale)
	h := int(float32(winH) * dprScale)

	// --- FBO PASS ---
	gl.BindFramebuffer(gl.FRAMEBUFFER, sceneFBO)
	gl.Viewport(0, 0, int32(w), int32(h))
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	gl.UseProgram(prog)

	proj := mgl32.Perspective(mgl32.DegToRad(45), winW/winH, 0.1, 5000)
	vp := proj.Mul4(getView())

	loc := gl.GetUniformLocation(prog, gl.Str("uVP\x00"))
	gl.UniformMatrix4fv(loc, 1, false, &vp[0])

	data := make([]float32, 0, nParticles*3)
	for _, p := range particles {
		data = append(data, p.Pos.X(), p.Pos.Y(), p.Pos.Z())
	}

	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(data)*4, gl.Ptr(data))

	gl.BindVertexArray(vao)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))

	// --- SCREEN PASS ---
	gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
	gl.Viewport(0, 0, int32(winW), int32(winH))
	gl.Clear(gl.COLOR_BUFFER_BIT)

	gl.UseProgram(screenProg)
	gl.BindTexture(gl.TEXTURE_2D, sceneTex)

	gl.BindVertexArray(quadVAO)
	gl.DrawArrays(gl.TRIANGLES, 0, 6)
}

//
// ---------------- FBO / QUAD ----------------
//

func setupFBO(w, h int) {
	gl.GenFramebuffers(1, &sceneFBO)
	gl.BindFramebuffer(gl.FRAMEBUFFER, sceneFBO)

	gl.GenTextures(1, &sceneTex)
	gl.BindTexture(gl.TEXTURE_2D, sceneTex)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, int32(w), int32(h), 0, gl.RGBA, gl.FLOAT, nil)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)

	gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTex, 0)

	gl.GenRenderbuffers(1, &depthRB)
	gl.BindRenderbuffer(gl.RENDERBUFFER, depthRB)
	gl.RenderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT, int32(w), int32(h))
	gl.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRB)

	if gl.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE {
		panic("FBO failed")
	}

	gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
}

func setupQuad() {
	quad := []float32{
		-1, -1, 0, 0,
		1, -1, 1, 0,
		1, 1, 1, 1,
		-1, -1, 0, 0,
		1, 1, 1, 1,
		-1, 1, 0, 1,
	}

	gl.GenVertexArrays(1, &quadVAO)
	gl.BindVertexArray(quadVAO)

	gl.GenBuffers(1, &quadVBO)
	gl.BindBuffer(gl.ARRAY_BUFFER, quadVBO)
	gl.BufferData(gl.ARRAY_BUFFER, len(quad)*4, gl.Ptr(quad), gl.STATIC_DRAW)

	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 2, gl.FLOAT, false, 4*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 2, gl.FLOAT, false, 4*4, gl.PtrOffset(2*4))
}

//
// ---------------- SHADERS ----------------
//

var particleVS = `
#version 410 core
layout(location=0) in vec3 pos;
uniform mat4 uVP;
void main() {
    gl_Position = uVP * vec4(pos,1.0);
    gl_PointSize = 4.0;
}` + "\x00"

var particleFS = `
#version 410 core
out vec4 fragColor;
void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if(d > 0.5) discard;
    fragColor = vec4(1,0.5,0.2,1);
}` + "\x00"

var screenVS = `
#version 410 core
layout(location=0) in vec2 pos;
layout(location=1) in vec2 uvIn;
out vec2 uv;
void main(){
    uv = uvIn;
    gl_Position = vec4(pos,0,1);
}` + "\x00"

var screenFS = `
#version 410 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D uTex;
void main(){
    fragColor = texture(uTex, uv);
}` + "\x00"

//
// ---------------- SHADER UTILS ----------------
//

func mustProgram(vs, fs string) uint32 {
	v := compile(vs, gl.VERTEX_SHADER)
	f := compile(fs, gl.FRAGMENT_SHADER)

	p := gl.CreateProgram()
	gl.AttachShader(p, v)
	gl.AttachShader(p, f)
	gl.LinkProgram(p)

	return p
}

func compile(src string, t uint32) uint32 {
	sh := gl.CreateShader(t)
	csrc, free := gl.Strs(src)
	gl.ShaderSource(sh, 1, csrc, nil)
	free()
	gl.CompileShader(sh)

	var status int32
	gl.GetShaderiv(sh, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		panic("shader compile failed")
	}
	return sh
}

//
// ---------------- DRS ----------------
//

func updateDRS(dt float64) {
	fps := 1.0 / dt

	if fps < targetFPS {
		dprScale -= 0.02
	}
	if fps > targetFPS+10 {
		dprScale += 0.02
	}

	if dprScale < 0.5 {
		dprScale = 0.5
	}
	if dprScale > 1.0 {
		dprScale = 1.0
	}
}
