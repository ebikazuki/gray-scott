uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    float r = texture2D(u_texture, v_texcoord).r;
    gl_FragColor = vec4(r*(1-0.03)+0.03,r*(1-0.08)+0.08,r*(1-0.4)+0.4,1);
}
