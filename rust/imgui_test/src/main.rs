use core::f32;
use std::f32::consts::PI;
use std::mem;
use std::os::raw::c_void;
use std::time::Duration;

use c_str_macro::c_str;
use cgmath::perspective;
use cgmath::prelude::SquareMatrix;
use gl::types::{GLfloat, GLsizei, GLsizeiptr};

use imgui::im_str;

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

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const FLOAT_NUM: usize = 3;
const VERTEX_NUM: usize = 36;
const BUF_LEN: usize = FLOAT_NUM * VERTEX_NUM;

fn draw_ui(ui: &imgui::Ui) {
    ui.text(im_str!("Hello World"));
    let mouse_pos = ui.io().mouse_pos;
    ui.text(format!("{:.1}, {:.1}", mouse_pos[0], mouse_pos[1]));

    imgui::ProgressBar::new(0.6)
        .size([200.0, 20.0])
        .overlay_text(im_str!("Progress"))
        .build(&ui);

    let arr = [0.6f32, 0.1f32, 1.0f32, 0.5f32, 0.92f32, 0.1f32, 0.2f32];
    ui.plot_lines(im_str!("lines"), &arr)
        .graph_size([200.0, 40.0])
        .build();

    ui.plot_histogram(im_str!("histogram"), &arr)
        .graph_size([200.0, 40.0])
        .build();
}

fn draw_effective_ui(ui: &imgui::Ui) {
    ui.text(im_str!("OpenGL Test App ver1.0"));
    ui.separator();
    ui.text(im_str!("FPS: {:.1}", ui.io().framerate));
    let display_size = ui.io().display_size;
    ui.text(format!(
        "Display Size: {:.1}, {:.1}",
        display_size[0], display_size[1]
    ));
    let mouse_pos = ui.io().mouse_pos;
    ui.text(format!(
        "Mouse Pos: ({:.1}, {:.1})",
        mouse_pos[0], mouse_pos[1]
    ));
}

fn draw_3d_ui(
    ui: &imgui::Ui,
    depth_test: &mut bool,
    blend: &mut bool,
    wireframe: &mut bool,
    culling: &mut bool,
    camera_x: &mut f32,
    camera_y: &mut f32,
    camera_z: &mut f32,
) {
    ui.text(im_str!("OpenGL Test App ver1.0"));
    ui.separator();
    ui.text(im_str!("FPS: {:.1}", ui.io().framerate));
    let display_size = ui.io().display_size;
    ui.text(format!(
        "Display Size: {:.1}, {:.1}",
        display_size[0], display_size[1]
    ));
    let mouse_pos = ui.io().mouse_pos;
    ui.text(format!(
        "Mouse Pos: ({:.1}, {:.1})",
        mouse_pos[0], mouse_pos[1]
    ));
    ui.separator();

    ui.checkbox(im_str!("Depth Test"), depth_test);
    ui.checkbox(im_str!("Blend"), blend);
    ui.checkbox(im_str!("Wireframe"), wireframe);
    ui.checkbox(im_str!("Culling"), culling);
    ui.separator();

    #[rustfmt::skip]
    imgui::Slider::new(im_str!("Camera X"), -5.0..=5.0)
        .build(&ui, camera_x);

    #[rustfmt::skip]
    imgui::Slider::new(im_str!("Camera Y"), -5.0..=5.0)
        .build(&ui, camera_y);

    #[rustfmt::skip]
    imgui::Slider::new(im_str!("Camera Z"), -5.0..=5.0)
        .build(&ui, camera_z);
}

fn set_opengl_version(video_subsystem: &sdl2::VideoSubsystem, major: u8, minor: u8) {
    let gl_attr = video_subsystem.gl_attr();
    gl_attr.set_context_profile(sdl2::video::GLProfile::Core);
    gl_attr.set_context_version(major, minor);
    let (major, minor) = gl_attr.context_version();
    println!("OK: init OpenGL: version={}.{}", major, minor);
}

fn init_imgui() -> imgui::Context {
    let mut imgui_context = imgui::Context::create();
    imgui_context.set_ini_filename(None);
    imgui_context
}

fn create_vertex() -> Vertex {
    #[rustfmt::skip]
    let buffer_array: [f32; BUF_LEN] = [
        // 1
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 1.0, 0.0,

        0.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 0.0, 0.0,

        // 2
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,

        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
        1.0, 0.0, 1.0,

        // 3
        0.0, 1.0, 1.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 1.0,

        0.0, 1.0, 1.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 1.0,

        // 4
        0.0, 1.0, 0.0,
        0.0, 1.0, 1.0,
        1.0, 1.0, 1.0,

        0.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 0.0,

        // 5
        1.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,

        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,

        // 6
        0.0, 1.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,

        0.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    //let buffer_array: [f32; BUF_LEN] = [
    //    -1.0, -1.0, 0.0,
    //    1.0, -1.0, 0.0,
    //    1.0, 1.0, 0.0,
    //];

    let vertex = Vertex::new(
        (BUF_LEN * mem::size_of::<GLfloat>()) as GLsizeiptr,
        buffer_array.as_ptr() as *const c_void,
        gl::DYNAMIC_DRAW,
        vec![gl::FLOAT],
        vec![FLOAT_NUM as i32],
        FLOAT_NUM as i32 * mem::size_of::<GLfloat>() as GLsizei,
        VERTEX_NUM as i32,
    );

    vertex
}

unsafe fn set_depth_test(depth_test: bool) {
    if depth_test {
        gl::Enable(gl::DEPTH_TEST);
    } else {
        gl::Disable(gl::DEPTH_TEST);
    }
}

unsafe fn set_blend(blend: bool) {
    if blend {
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    } else {
        gl::Disable(gl::BLEND);
    }
}

unsafe fn set_wireframe(wireframe: bool) {
    if wireframe {
        gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
    } else {
        gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
    }
}

unsafe fn set_culling(culling: bool) {
    if culling {
        gl::Enable(gl::CULL_FACE);
    } else {
        gl::Disable(gl::CULL_FACE);
    }
}

unsafe fn create_model_matrix(counter: &i32) -> Matrix4 {
    //let model_matrix = Matrix4::identity();
    let model_matrix =
        Matrix4::from_angle_z(cgmath::Rad(*counter as f32 / 360.0 * 2. * f32::consts::PI));
    //let model_matrix = Matrix4::from_angle_z(cgmath::Rad(f32::consts::PI));
    //        * Matrix4::from_scale(0.01 * counter as f32 + 0.1);
    //let fps = ui.io().framerate;
    //model_matrix = model_matrix
    //* Matrix4::from_angle_z(cgmath::Rad(
    //0.1 * counter as f32 * fps / 360.0 * 2. * f32::consts::PI,
    //));
    model_matrix
}

unsafe fn create_view_matrix(camera_x: f32, camera_y: f32, camera_z: f32) -> Matrix4 {
    Matrix4::look_at(
        Point3 {
            x: camera_x,
            y: camera_y,
            z: camera_z,
            //x: (counter as f32 / 360.0 * 2. * PI).cos(),
            //y: (counter as f32 / 360.0 * 2. * PI).sin(),
            //x: 0.0,
            //y: 0.0,
            //z: 3.0,
        },
        Point3 {
            //x: (counter as f32 / 360.0 * 2. * PI).cos(),
            //y: (counter as f32 / 360.0 * 2. * PI).sin(),
            x: 0.5,
            y: 0.5,
            z: 0.5,
        },
        Vector3 {
            //x: (counter as f32 / 360.0 * 2. * PI).cos(),
            //y: (counter as f32 / 360.0 * 2. * PI).sin(),
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
    )
}

unsafe fn create_projection_matrix() -> Matrix4 {
    perspective(
        //cgmath::Deg(1. + counter as f32),
        cgmath::Deg(45.0),
        WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32,
        0.1,
        100.0,
    )
}

//unsafe fn set_imgui_frame(
//    ui: &imgui::Ui,
//    imgui_sdl2_context: &mut imgui_sdl2::ImguiSdl2,
//    window: &sdl2::video::Window,
//    event_pump: &sdl2::EventPump,
//    imgui_context: &mut imgui::Context,
//) {
//    imgui_sdl2_context.prepare_frame(imgui_context.io_mut(), &window, &event_pump.mouse_state());
//    imgui::Window::new(im_str!("Information"))
//        .size([300.0, 200.0], imgui::Condition::FirstUseEver)
//        .build(ui, || draw_effective_ui(ui));
//    imgui_sdl2_context.prepare_render(ui, &window);
//}

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    set_opengl_version(&video_subsystem, 3, 1);

    let window = video_subsystem
        .window("SDL", WINDOW_WIDTH, WINDOW_HEIGHT)
        .opengl()
        .position_centered()
        .build()
        .unwrap();

    let _gl_context = window.gl_create_context().unwrap();
    gl::load_with(|s| video_subsystem.gl_get_proc_address(s) as _);

    let shader = Shader::new("rsc/shader/shader.vs", "rsc/shader/shader.fs");
    let vertex = create_vertex();

    let mut depth_test: bool = true;
    let mut blend: bool = true;
    let mut wireframe: bool = true;
    let mut culling: bool = true;
    let mut camera_x: f32 = 5.0f32;
    let mut camera_y: f32 = -5.0f32;
    //let mut camera_x: f32 = 0.0f32;
    //let mut camera_y: f32 = 0.0f32;
    let mut camera_z: f32 = 5.0f32;

    // init imgui
    let mut imgui_context = init_imgui();

    // init imgui sdl2
    let mut imgui_sdl2_context = imgui_sdl2::ImguiSdl2::new(&mut imgui_context, &window);
    let renderer = imgui_opengl_renderer::Renderer::new(&mut imgui_context, |s| {
        video_subsystem.gl_get_proc_address(s) as _
    });

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut counter = 0;
    'running: loop {
        for event in event_pump.poll_iter() {
            imgui_sdl2_context.handle_event(&mut imgui_context, &event);
            if imgui_sdl2_context.ignore_event(&event) {
                continue;
            }

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
            // set bools
            set_depth_test(depth_test);
            set_blend(blend);
            set_wireframe(wireframe);
            set_culling(culling);

            // view initialization
            gl::Viewport(0, 0, WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32);
            // clear screen
            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            // init matrice for model view and projection
            let model_matrix = create_model_matrix(&counter);
            let view_matrix = create_view_matrix(camera_x, camera_y, camera_z);
            let projection_matrix = create_projection_matrix();

            // shader use matrices
            shader.use_program();
            shader.set_mat4(c_str!("uModel"), &model_matrix);
            shader.set_mat4(c_str!("uView"), &view_matrix);
            shader.set_mat4(c_str!("uProjection"), &projection_matrix);
            vertex.draw();

            imgui_sdl2_context.prepare_frame(
                imgui_context.io_mut(),
                &window,
                &event_pump.mouse_state(),
            );

            //imgui
            //set_imgui_frame(
            //    &ui,
            //    &mut imgui_sdl2_context,
            //    &window,
            //    &event_pump,
            //    &mut imgui_context,
            //);

            let ui = imgui_context.frame();
            imgui::Window::new(im_str!("Information"))
                .size([300.0, 200.0], imgui::Condition::FirstUseEver)
                .build(&ui, || {
                    draw_3d_ui(
                        &ui,
                        &mut depth_test,
                        &mut blend,
                        &mut wireframe,
                        &mut culling,
                        &mut camera_x,
                        &mut camera_y,
                        &mut camera_z,
                    )
                });

            imgui_sdl2_context.prepare_render(&ui, &window);
            renderer.render(ui);
            window.gl_swap_window();
        }
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        counter += 1;
    }
}
