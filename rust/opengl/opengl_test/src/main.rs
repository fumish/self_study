use core::f32;
use std::f32::consts::PI;
use std::mem;
use std::os::raw::c_void;
use std::time::Duration;

use c_str_macro::c_str;
use cgmath::perspective;
use cgmath::prelude::SquareMatrix;
use gl::types::{GLfloat, GLsizei, GLsizeiptr};

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

mod shader;
mod vertex;

use shader::Shader;
use vertex::Vertex;

#[allow(dead_code)]
type Point3 = cgmath::Point3<f32>;
#[allow(dead_code)]
type Vector3 = cgmath::Vector3<f32>;
#[allow(dead_code)]
type Matrix4 = cgmath::Matrix4<f32>;

const WINDOW_WIDTH: u32 = 640;
const WINDOW_HEIGHT: u32 = 480;
const FLOAT_NUM: usize = 3;
const VERTEX_NUM: usize = 3;
const BUF_LEN: usize = FLOAT_NUM * VERTEX_NUM;

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    {
        let gl_attr = video_subsystem.gl_attr();
        gl_attr.set_context_profile(sdl2::video::GLProfile::Core);
        gl_attr.set_context_version(3, 1);
        let (major, minor) = gl_attr.context_version();
        println!("OK: init OpenGL: version={}.{}", major, minor);
    }

    let window = video_subsystem
        .window("SDL", WINDOW_WIDTH, WINDOW_HEIGHT)
        .opengl()
        .position_centered()
        .build()
        .unwrap();

    let _gl_context = window.gl_create_context().unwrap();
    gl::load_with(|s| video_subsystem.gl_get_proc_address(s) as _);

    let shader = Shader::new("rsc/shader/shader.vs", "rsc/shader/shader.fs");

    // set buffer
    #[rustfmt::skip]
    let buffer_array: [f32; BUF_LEN] = [
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        1.0, 1.0, 0.0,
    ];

    let vertex = Vertex::new(
        (BUF_LEN * mem::size_of::<GLfloat>()) as GLsizeiptr,
        buffer_array.as_ptr() as *const c_void,
        gl::DYNAMIC_DRAW,
        vec![gl::FLOAT],
        vec![FLOAT_NUM as i32],
        FLOAT_NUM as i32 * mem::size_of::<GLfloat>() as GLsizei,
        VERTEX_NUM as i32,
    );
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut counter = 0;
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        unsafe {
            gl::Viewport(0, 0, WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32);
            // clear screen
            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            // init matrice for model view and projection
            let model_matrix = Matrix4::identity();
            //let model_matrix = Matrix4::from_angle_z(cgmath::Rad(f32::consts::PI));
            let view_matrix = Matrix4::look_at(
                Point3 {
                    x: (counter as f32 / 360.0 * 2. * PI).cos(),
                    y: (counter as f32 / 360.0 * 2. * PI).sin(),
                    //x: 0.0,
                    //y: 0.0,
                    z: 3.0,
                },
                Point3 {
                    //x: (counter as f32 / 360.0 * 2. * PI).cos(),
                    //y: (counter as f32 / 360.0 * 2. * PI).sin(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vector3 {
                    //x: (counter as f32 / 360.0 * 2. * PI).cos(),
                    //y: (counter as f32 / 360.0 * 2. * PI).sin(),
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            );
            let projection_matrix: Matrix4 = perspective(
                //cgmath::Deg(1. + counter as f32),
                cgmath::Deg(45.0),
                WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32,
                0.1,
                100.0,
            );
            //println!("projection mat: {:?}", projection_matrix);

            // shader use matrices
            shader.use_program();
            shader.set_mat4(c_str!("uModel"), &model_matrix);
            shader.set_mat4(c_str!("uView"), &view_matrix);
            shader.set_mat4(c_str!("uProjection"), &projection_matrix);
            vertex.draw();
            window.gl_swap_window();
        }
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        counter += 1;
    }
}

//     let mut event_pump = sdl_context.event_pump().unwrap();
//
//     let mut canvas = window.into_canvas().build().unwrap();
//     canvas.set_draw_color(Color::RGB(255, 255, 255));
//     canvas.clear();
//     canvas.present();
//
//     let mut event_pump = sdl_context.event_pump().unwrap();
// }
//
