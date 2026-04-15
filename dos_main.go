package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

const (
	winWidth  = 1280
	winHeight = 720

	nParticles = 2500 // Increased count for better visual
	numWorkers = 8

	sphereRadius   = 300.0
	springStrength = 45.0
)

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

var (
	particles []Particle

	prog     uint32
	quadProg uint32

	vao uint32
	vbo uint32

	quadVAO uint32
	quadVBO uint32

	sceneFBO uint32
	sceneTex uint32
	depthRB  uint32

	timeAccumulator float32

	azimuth, elevation float64 = 0.5, 0.2
	distance           float64 = 800

	winW = float32(winWidth)
	winH = float32(winHeight)

	mouseDown bool
	lastX     float64
	lastY     float64
)

func init() {
	runtime.LockOSThread()
}

func main() {
	rand.Seed(time.Now().UnixNano())

	if err := glfw.Init(); err != nil {
		log.Fatal(err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, gl.TRUE)

	window, err := glfw.CreateWindow(winWidth, winHeight, "Glow Sphere - Corrected", nil, nil)
	if err != nil {
		panic(err)
	}
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(err)
	}

	// General GL State
	gl.Enable(gl.PROGRAM_POINT_SIZE)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE)
	gl.Enable(gl.DEPTH_TEST)

	// Build Shaders
	prog = mustProgram(particleVS, particleFS)
	quadProg = mustProgram(quadVS, quadFS)

	initParticles()
	setupBuffers()
	setupQuad()
	setupFBOs()

	window.SetCursorPosCallback(mouseMove)
	window.SetMouseButtonCallback(mouseButton)

	prev := time.Now()

	for !window.ShouldClose() {
		now := time.Now()
		dt := now.Sub(prev).Seconds()
		prev = now

		// Physics
		update(dt)

		// Pass 1: Render Particles to FBO
		renderScene()

		// Pass 2: Draw FBO Texture to Screen
		renderComposite()

		window.SwapBuffers()
		glfw.PollEvents()
	}
}

func initParticles() {
	particles = make([]Particle, nParticles)
	for i := range particles {
		a := rand.Float64() * 2 * math.Pi
		b := rand.Float64() * math.Pi

		particles[i].Pos = mgl32.Vec3{
			sphereRadius * float32(math.Cos(a)*math.Sin(b)),
			sphereRadius * float32(math.Sin(a)*math.Sin(b)),
			sphereRadius * float32(math.Cos(b)),
		}
		particles[i].Col = mgl32.Vec3{1.0, 0.5, 0.2}
	}
}

func update(dt float64) {
	timeAccumulator += float32(dt)
	center := spirograph(timeAccumulator)

	var wg sync.WaitGroup
	chunk := len(particles) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start, end := w*chunk, (w+1)*chunk
		if w == numWorkers-1 {
			end = len(particles)
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fdt := float32(dt)
			for i := s; i < e; i++ {
				p := &particles[i]
				dir := p.Pos.Sub(center)
				dist := dir.Len()

				if dist > 0 {
					force := dir.Normalize().Mul(-float32((dist - sphereRadius) * springStrength))
					p.Vel = p.Vel.Add(force.Mul(fdt))
				}

				p.Pos = p.Pos.Add(p.Vel.Mul(fdt))
				p.Vel = p.Vel.Mul(0.95) // Damping

				speed := p.Vel.Len()
				p.Col = mgl32.Vec3{
					0.5 + speed*0.02,
					0.2 + speed*0.01,
					0.8 + speed*0.05,
				}
			}
		}(start, end)
	}
	wg.Wait()
}

func spirograph(t float32) mgl32.Vec3 {
	R, k, l := float32(180), float32(0.531), float32(0.854)
	ratio := (1 - k) / k
	x := R*(1-k)*float32(math.Cos(float64(t))) + R*l*k*float32(math.Cos(float64(ratio*t)))
	y := R*(1-k)*float32(math.Sin(float64(t))) - R*l*k*float32(math.Sin(float64(ratio*t)))
	z := R * 0.5 * float32(math.Sin(float64(t*0.3)))
	return mgl32.Vec3{x, y, z}
}

func renderScene() {
	gl.BindFramebuffer(gl.FRAMEBUFFER, sceneFBO)
	gl.Viewport(0, 0, winWidth, winHeight)
	gl.ClearColor(0, 0, 0, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	gl.UseProgram(prog)

	proj := mgl32.Perspective(mgl32.DegToRad(45), winW/winH, 0.1, 5000)
	view := camera()
	vp := proj.Mul4(view)

	loc := gl.GetUniformLocation(prog, gl.Str("uVP\x00"))
	gl.UniformMatrix4fv(loc, 1, false, &vp[0])

	data := make([]float32, 0, nParticles*6)
	for _, p := range particles {
		data = append(data, p.Pos.X(), p.Pos.Y(), p.Pos.Z(), p.Col.X(), p.Col.Y(), p.Col.Z())
	}

	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(data)*4, gl.Ptr(data))

	gl.BindVertexArray(vao)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))
}

func renderComposite() {
	gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
	gl.Disable(gl.DEPTH_TEST)

	gl.UseProgram(quadProg)
	gl.ActiveTexture(gl.TEXTURE0)
	gl.BindTexture(gl.TEXTURE_2D, sceneTex)
	gl.Uniform1i(gl.GetUniformLocation(quadProg, gl.Str("uTex\x00")), 0)

	gl.BindVertexArray(quadVAO)
	gl.DrawArrays(gl.TRIANGLES, 0, 6)
	gl.Enable(gl.DEPTH_TEST)
}

func setupBuffers() {
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)

	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferData(gl.ARRAY_BUFFER, nParticles*6*4, nil, gl.DYNAMIC_DRAW)

	gl.EnableVertexAttribArray(0) // pos
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))
	gl.EnableVertexAttribArray(1) // col
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))
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

func setupFBOs() {
	gl.GenFramebuffers(1, &sceneFBO)
	gl.BindFramebuffer(gl.FRAMEBUFFER, sceneFBO)

	gl.GenTextures(1, &sceneTex)
	gl.BindTexture(gl.TEXTURE_2D, sceneTex)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, winWidth, winHeight, 0, gl.RGBA, gl.FLOAT, nil)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTex, 0)

	gl.GenRenderbuffers(1, &depthRB)
	gl.BindRenderbuffer(gl.RENDERBUFFER, depthRB)
	gl.RenderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT, winWidth, winHeight)
	gl.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRB)

	if gl.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE {
		panic("FBO Incomplete")
	}
	gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
}

func camera() mgl32.Mat4 {
	x := float32(distance * math.Cos(azimuth) * math.Cos(elevation))
	y := float32(distance * math.Sin(elevation))
	z := float32(distance * math.Sin(azimuth) * math.Cos(elevation))
	return mgl32.LookAtV(mgl32.Vec3{x, y, z}, mgl32.Vec3{0, 0, 0}, mgl32.Vec3{0, 1, 0})
}

func mouseMove(w *glfw.Window, x, y float64) {
	if mouseDown {
		azimuth -= (x - lastX) * 0.005
		elevation -= (y - lastY) * 0.005
		if elevation > 1.5 {
			elevation = 1.5
		}
		if elevation < -1.5 {
			elevation = -1.5
		}
	}
	lastX, lastY = x, y
}

func mouseButton(w *glfw.Window, b glfw.MouseButton, a glfw.Action, m glfw.ModifierKey) {
	mouseDown = b == glfw.MouseButtonLeft && a == glfw.Press
}

// ---------------- SHADERS ----------------

var particleVS = `
#version 410 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inCol;
uniform mat4 uVP;
out vec3 vCol;
void main(){
    gl_Position = uVP * vec4(inPos,1.0);
    vCol = inCol;
    gl_PointSize = 4.0 + (500.0 / gl_Position.w);
}` + "\x00"

var particleFS = `
#version 410 core
in vec3 vCol;
out vec4 fragColor;
void main(){
    float d = length(gl_PointCoord - vec2(0.5));
    if(d > 0.5) discard;
    float a = smoothstep(0.5, 0.1, d);
    fragColor = vec4(vCol, a);
}` + "\x00"

var quadVS = `
#version 410 core
layout(location=0) in vec2 pos;
layout(location=1) in vec2 uvIn;
out vec2 uv;
void main(){
    uv = uvIn;
    gl_Position = vec4(pos, 0, 1);
}` + "\x00"

var quadFS = `
#version 410 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D uTex;
void main(){
    vec3 col = texture(uTex, uv).rgb;
    // Simple bloom-ish reinforcement
    fragColor = vec4(col * 1.2, 1.0);
}` + "\x00"

// ---------------- UTILS ----------------

func mustProgram(vs, fs string) uint32 {
	v := compile(vs, gl.VERTEX_SHADER)
	f := compile(fs, gl.FRAGMENT_SHADER)
	p := gl.CreateProgram()
	gl.AttachShader(p, v)
	gl.AttachShader(p, f)
	gl.LinkProgram(p)

	var status int32
	gl.GetProgramiv(p, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetProgramiv(p, gl.INFO_LOG_LENGTH, &logLength)
		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetProgramInfoLog(p, logLength, nil, gl.Str(log))
		panic(fmt.Errorf("failed to link program: %v", log))
	}
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
		var logLength int32
		gl.GetShaderiv(sh, gl.INFO_LOG_LENGTH, &logLength)
		log := strings.Repeat("\x00", int(logLength+1))
		gl.GetShaderInfoLog(sh, logLength, nil, gl.Str(log))
		panic(fmt.Errorf("failed to compile %v: %v", t, log))
	}
	return sh
}
