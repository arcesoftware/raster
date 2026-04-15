package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"github.com/go-gl/gl/v4.1-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/go-gl/mathgl/mgl32"
)

const (
	winWidth   = 1280
	winHeight  = 720
	nParticles = 1618

	numWorkers = 8

	pointBaseSize = 3.0

	sphereRadius   = 150.0
	springStrength = 22.0

	spiroR     float32 = 180.0
	spiroK     float32 = 0.531
	spiroL     float32 = 0.854
	spiroSpeed float32 = 0.8
)

type Particle struct {
	Pos mgl32.Vec3
	Vel mgl32.Vec3
	Col mgl32.Vec3
}

var (
	particles       []Particle
	timeAccumulator float32

	prog uint32
	vao  uint32
	vbo  uint32

	azimuth, elevation float64 = 0.6, 0.2
	distance           float64 = 700
)

func init() {
	runtime.LockOSThread()
}

func main() {
	rand.Seed(time.Now().UnixNano())

	if err := glfw.Init(); err != nil {
		log.Fatalln(err)
	}
	defer glfw.Terminate()

	glfw.WindowHint(glfw.ContextVersionMajor, 4)
	glfw.WindowHint(glfw.ContextVersionMinor, 1)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)

	window, _ := glfw.CreateWindow(winWidth, winHeight, "Glow Particles", nil, nil)
	window.MakeContextCurrent()

	if err := gl.Init(); err != nil {
		panic(err)
	}

	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE) // additive glow
	gl.Enable(gl.PROGRAM_POINT_SIZE)

	prog, _ = newProgram(vertexShader, fragmentShader)

	initParticles()
	setupBuffers()

	prev := time.Now()

	for !window.ShouldClose() {
		now := time.Now()
		dt := now.Sub(prev).Seconds()
		prev = now

		update(dt)
		render()

		window.SwapBuffers()
		glfw.PollEvents()
	}
}

func initParticles() {
	particles = make([]Particle, nParticles)

	for i := range particles {
		angle1 := rand.Float64() * 2 * math.Pi
		angle2 := rand.Float64() * math.Pi

		r := sphereRadius + float32(rand.Float64()*10-5)

		pos := mgl32.Vec3{
			r * float32(math.Cos(angle1)) * float32(math.Sin(angle2)),
			r * float32(math.Sin(angle1)) * float32(math.Sin(angle2)),
			r * float32(math.Cos(angle2)),
		}

		particles[i] = Particle{
			Pos: pos,
			Vel: mgl32.Vec3{
				float32(rand.Float64()*2 - 1),
				float32(rand.Float64()*2 - 1),
				float32(rand.Float64()*2 - 1),
			},
			Col: mgl32.Vec3{1, 0.3, 0.2},
		}
	}
}

func update(dt float64) {
	timeAccumulator += float32(dt)

	// subtle camera motion
	azimuth += 0.1 * dt

	center := spirograph(timeAccumulator)

	var wg sync.WaitGroup
	chunk := len(particles) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunk
		end := start + chunk
		if w == numWorkers-1 {
			end = len(particles)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			for i := start; i < end; i++ {
				p := &particles[i]

				dir := p.Pos.Sub(center)
				dist := dir.Len()

				if dist > 0 {
					force := dir.Normalize().Mul(-float32((dist - sphereRadius) * springStrength))
					p.Vel = p.Vel.Add(force.Mul(float32(dt)))
				}

				p.Pos = p.Pos.Add(p.Vel.Mul(float32(dt)))
				p.Vel = p.Vel.Mul(0.96)

				// velocity-based color
				speed := p.Vel.Len()
				p.Col = mgl32.Vec3{
					0.8 + speed*0.05,
					0.3 + speed*0.02,
					0.2 + speed*0.08,
				}
			}
		}(start, end)
	}

	wg.Wait()
}

func spirograph(t float32) mgl32.Vec3 {
	R := spiroR
	k := spiroK
	l := spiroL

	ratio := (1 - k) / k

	x := R*(1-k)*float32(math.Cos(float64(t))) +
		R*l*k*float32(math.Cos(float64(ratio*t)))

	y := R*(1-k)*float32(math.Sin(float64(t))) -
		R*l*k*float32(math.Sin(float64(ratio*t)))

	z := R * 0.2 * float32(math.Sin(float64(t*0.5)))

	return mgl32.Vec3{x, y, z}
}

func setupBuffers() {
	gl.GenVertexArrays(1, &vao)
	gl.BindVertexArray(vao)

	gl.GenBuffers(1, &vbo)
	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)

	gl.BufferData(gl.ARRAY_BUFFER, nParticles*6*4, nil, gl.DYNAMIC_DRAW)

	gl.EnableVertexAttribArray(0)
	gl.VertexAttribPointer(0, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(0))

	gl.EnableVertexAttribArray(1)
	gl.VertexAttribPointer(1, 3, gl.FLOAT, false, 6*4, gl.PtrOffset(3*4))
}

func render() {
	gl.ClearColor(0.02, 0.02, 0.05, 1)
	gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

	gl.UseProgram(prog)

	proj := mgl32.Perspective(mgl32.DegToRad(45), float32(winWidth)/winHeight, 0.1, 5000)
	view := cameraMatrix()
	vp := proj.Mul4(view)

	loc := gl.GetUniformLocation(prog, gl.Str("uVP\x00"))
	gl.UniformMatrix4fv(loc, 1, false, &vp[0])

	data := make([]float32, 0, nParticles*6)
	for _, p := range particles {
		data = append(data, p.Pos.X(), p.Pos.Y(), p.Pos.Z())
		data = append(data, p.Col.X(), p.Col.Y(), p.Col.Z())
	}

	gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
	gl.BufferSubData(gl.ARRAY_BUFFER, 0, len(data)*4, gl.Ptr(data))

	gl.BindVertexArray(vao)
	gl.DrawArrays(gl.POINTS, 0, int32(nParticles))
}

func cameraMatrix() mgl32.Mat4 {
	x := float32(distance * math.Cos(azimuth) * math.Cos(elevation))
	y := float32(distance * math.Sin(elevation))
	z := float32(distance * math.Sin(azimuth) * math.Cos(elevation))

	return mgl32.LookAtV(
		mgl32.Vec3{x, y, z},
		mgl32.Vec3{0, 0, 0},
		mgl32.Vec3{0, 1, 0},
	)
}

// ---------- SHADERS ----------

var vertexShader = `
#version 410 core
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inCol;

uniform mat4 uVP;

out vec3 vColor;

void main() {
    gl_Position = uVP * vec4(inPos, 1.0);
    vColor = inCol;

    float size = 3.0 + length(inPos) * 0.002;
    gl_PointSize = size;
}
` + "\x00"

var fragmentShader = `
#version 410 core

in vec3 vColor;
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    float alpha = smoothstep(0.5, 0.0, dist);

    vec3 color = vColor * (1.5 - dist);

    fragColor = vec4(color, alpha);
}
` + "\x00"

// ---------- SHADER UTILS ----------

func compileShader(src string, t uint32) (uint32, error) {
	shader := gl.CreateShader(t)
	csources, free := gl.Strs(src)
	gl.ShaderSource(shader, 1, csources, nil)
	free()
	gl.CompileShader(shader)

	var status int32
	gl.GetShaderiv(shader, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		var logLength int32
		gl.GetShaderiv(shader, gl.INFO_LOG_LENGTH, &logLength)
		log := string(make([]byte, logLength+1))
		return 0, fmt.Errorf("shader error: %v", log)
	}
	return shader, nil
}

func newProgram(vs, fs string) (uint32, error) {
	v, _ := compileShader(vs, gl.VERTEX_SHADER)
	f, _ := compileShader(fs, gl.FRAGMENT_SHADER)

	prog := gl.CreateProgram()
	gl.AttachShader(prog, v)
	gl.AttachShader(prog, f)
	gl.LinkProgram(prog)

	return prog, nil
}
