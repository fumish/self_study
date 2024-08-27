use cgmath::Array;
use cgmath::Matrix;
use gl;
use gl::types::*;

use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::Read;
use std::ptr;
use std::str;

#[allow(dead_code)]
type Vector3 = cgmath::Vector3<f32>;

#[allow(dead_code)]
type Matrix4 = cgmath::Matrix4<f32>;

pub struct Shader {
    pub id: u32,
}
#[allow(dead_code)]
impl Shader {
    #[rustfmt::skip]
    pub fn new(vertex_path: &str, fragment_path: &str) -> Shader {
        let mut shader = Shader{id: 0};
        let cstr_vertex_code = Shader::path2_cstr_code(vertex_path);
        let cstr_fragment_code = Shader::path2_cstr_code(fragment_path);

        unsafe {
            //vertex shader
            let vertex = shader.compile_shader(cstr_vertex_code, gl::VERTEX_SHADER, "VERTEX");

            //fragment shader
            let fragment = shader.compile_shader(cstr_fragment_code, gl::FRAGMENT_SHADER, "FRAGMENT");

            // shader program
            let id = Shader::create_program(vec![vertex, fragment]);
            shader.id = id;
        }
        shader
    }

    pub unsafe fn use_program(&self) {
        gl::UseProgram(self.id)
    }

    pub unsafe fn set_bool(&self, name: &CStr, value: bool) {
        gl::Uniform1i(gl::GetUniformLocation(self.id, name.as_ptr()), value as i32);
    }

    pub unsafe fn set_int(&self, name: &CStr, value: i32) {
        gl::Uniform1i(gl::GetUniformLocation(self.id, name.as_ptr()), value);
    }

    pub unsafe fn set_float(&self, name: &CStr, value: f32) {
        gl::Uniform1f(gl::GetUniformLocation(self.id, name.as_ptr()), value);
    }

    pub unsafe fn set_vector3(&self, name: &CStr, value: &Vector3) {
        gl::Uniform3fv(
            gl::GetUniformLocation(self.id, name.as_ptr()),
            1,
            value.as_ptr(),
        );
    }

    pub unsafe fn set_vec3(&self, name: &CStr, x: f32, y: f32, z: f32) {
        gl::Uniform3f(gl::GetUniformLocation(self.id, name.as_ptr()), x, y, z);
    }

    pub unsafe fn set_mat4(&self, name: &CStr, value: &Matrix4) {
        gl::UniformMatrix4fv(
            gl::GetUniformLocation(self.id, name.as_ptr()),
            1,
            gl::FALSE,
            value.as_ptr(),
        );
    }

    unsafe fn check_compile_errors(&self, shader: u32, type_: &str) {
        let mut success = gl::FALSE as GLint;
        let mut info_log = Vec::with_capacity(1024);
        info_log.set_len(1024 - 1);

        if type_ != "PROGRAM" {
            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl::GetShaderInfoLog(
                    shader,
                    1024,
                    ptr::null_mut(),
                    info_log.as_mut_ptr() as *mut GLchar,
                );

                let info_log_string = match String::from_utf8(info_log) {
                    Ok(log) => log,
                    Err(vec) => panic!("Failed to compilation log from buffer: {}", vec),
                };

                println!(
                    "failed to compile shader code: type={}, log={}",
                    type_, info_log_string
                );
            }
        } else {
            gl::GetProgramiv(shader, gl::LINK_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl::GetProgramInfoLog(
                    shader,
                    1024,
                    ptr::null_mut(),
                    info_log.as_mut_ptr() as *mut GLchar,
                );

                let info_log_string = match String::from_utf8(info_log) {
                    Ok(log) => log,
                    Err(vec) => panic!("Failed to convert to link log from buffer {}", vec),
                };

                println!(
                    "failed to link shader code: type={}, log={}",
                    type_, info_log_string
                );
            }
        }
    }
    #[rustfmt::skip]
    pub fn with_geometry_shader(
        vertex_path: &str,
        fragment_path: &str,
        geometry_path: &str,
    ) -> Self {
        let mut shader = Shader{id: 0};
        let cstr_vertex_code = Shader::path2_cstr_code(vertex_path);
        let cstr_fragment_code = Shader::path2_cstr_code(fragment_path);
        let cstr_geometry_code = Shader::path2_cstr_code(geometry_path);

        unsafe {
            //vertex shader
            let vertex = shader.compile_shader(cstr_vertex_code, gl::VERTEX_SHADER, "VERTEX");

            //fragment shader
            let fragment = shader.compile_shader(cstr_fragment_code, gl::FRAGMENT_SHADER, "FRAGMENT");

            //geometry shader
            let geometry = shader.compile_shader(cstr_geometry_code, gl::GEOMETRY_SHADER, "GEOMETRY");

            //shader program
            let id = Shader::create_program(vec![vertex, fragment, geometry]);

            gl::DeleteProgram(id);
            shader.id = id;
        }
        shader
    }

    /// Create shader program
    /// # Safety
    /// This function is unsafe because it calls OpenGL functions.
    /// # Examples
    /// ```
    /// let objects = vec![1, 2, 3];
    /// let id = Shader::crate_program(objects);
    /// ```
    /// # Panics
    /// This function panics if the program cannot be created.
    ///
    unsafe fn create_program(objects: Vec<u32>) -> u32 {
        let id = gl::CreateProgram();
        for object in objects.iter() {
            gl::AttachShader(id, *object);
        }
        gl::LinkProgram(id);

        for object in objects {
            gl::DeleteShader(object);
        }
        id
    }

    /// Convert file path to CString
    /// # Panics
    /// This function panics if the file does not exist or if the file cannot be read.
    /// # Examples
    /// ```
    /// let path = "path/to/file";
    /// let cstr_code = Shader::path2_cstr_code(path);
    /// ```
    fn path2_cstr_code(path: &str) -> CString {
        let mut path_file =
            File::open(path).unwrap_or_else(|_| panic!("Failed to open file: {}", path));

        let mut path_code = String::new();

        path_file
            .read_to_string(&mut path_code)
            .expect("failed to read path file.");

        CString::new(path_code.as_bytes()).unwrap()
    }

    /// Compile shader code
    /// # Safety
    /// This function is unsafe because it calls OpenGL functions.
    /// # Examples
    /// ```
    /// let path = "path/to/vertex_file";
    /// let cstr_vertex_code = Shader::path2_cstr_code(path);
    /// let shader_enum = gl::VERTEX_SHADER;
    /// let shader_str = "VERTEX";
    /// let shader = shader.compile_shader(shader_code, shader_enum, shader_str);
    /// ```
    ///
    unsafe fn compile_shader(
        &self,
        shader_code: CString,
        shader_enum: GLenum,
        shader_str: &str,
    ) -> u32 {
        let shader = gl::CreateShader(shader_enum);
        gl::ShaderSource(shader, 1, &shader_code.as_ptr(), ptr::null());
        gl::CompileShader(shader);
        self.check_compile_errors(shader, shader_str);
        shader
    }
}
