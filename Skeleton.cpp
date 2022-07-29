//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kászonyi Zsombor	
// Neptun : DCE2Q1
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

float myT  = M_PI / 11;
const float epsilon = 0.0001f;
vec3 firRD	   = vec3(0.0f,    4.0f, 1.0f);
vec3 secRD	   = vec3(2.0f,    5.0f, 1.0f);
vec3 thiRD	   = vec3(0.6f, -  0.7f, 0.6f);
vec3 rotOrigin = vec3(0.0f, -0.325f, 0.0f);

vec4 quat(vec3 a, float angle) {
	return vec4(a.x * sin(angle / 2), a.y * sin(angle / 2), a.z * sin(angle / 2), cos(angle / 2));
}

vec4 quatInv(vec4 q) {
	return vec4(-q.x, -q.y, -q.z, q.w);
}

vec4 quatMul(vec4 q1, vec4 q2) {
	vec3 temp1 = vec3(q1.x, q1.y, q1.z);
	vec3 temp2 = vec3(q2.x, q2.y, q2.z);
	vec3 re(q1.w * temp2 + q2.w * temp1 + cross(temp1, temp2));
	return vec4(re.x, re.y, re.z, q1.w * q2.w - dot(temp1, temp2));
}

vec3 quatRot(vec4 q, vec3 p) {
	vec4 qInv = quatInv(q);
	vec4 re = quatMul(quatMul(q, vec4(p.x, p.y, p.z, 0)), qInv);
	return vec3(re.x, re.y, re.z);
}

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	vec3 center;
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	float radius;
	int layer;
	Sphere* below, *second;


	Sphere(const vec3& _center, float _radius, Material* _material, int _layer, Sphere* b = nullptr, Sphere* sec = nullptr) {
		center = _center;
		radius = _radius;
		material = _material;
		layer = _layer;
		below = b;
		second = sec;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec4 q1 = quat(normalize(firRD), myT);
		vec4 q2 = quat(normalize(secRD), myT);
		vec3 s  = ray.start, d = ray.dir, nC = center;
		if (layer > 0) {
			s = quatRot(q1, ray.start - rotOrigin);
			d = quatRot(q1, ray.dir);
			nC = center - rotOrigin;
			
			if (layer > 1) {
				s = quatRot(q2, s - below->center + rotOrigin);
				d = quatRot(q2, d);
				nC = nC - below->center + rotOrigin;
			}
		}
		vec3 dist = vec3(s - nC);
		float a = dot(d, d);
		float b = dot(dist, d) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = s + d * hit.t;
		hit.normal = (hit.position - nC) / radius;
		if (layer > 0) {
			if (layer > 1)
				hit.normal = quatRot(quatInv(q2), hit.normal);
			hit.normal = quatRot(quatInv(q1), hit.normal);
		}
		hit.material = material;
		return hit;
	}
};

struct Plain : public Intersectable {
	vec3 normal;

	Plain(vec3 cen, vec3 n, Material* mat) {
		center = cen; normal = n; material = mat;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 s = ray.start - center;
		vec3 d = ray.dir;
		hit.t = dot(center - s, normal) / dot(d, normal);
		hit.normal = normal;
		hit.position = s + d * hit.t;
		hit.material = material;
		return hit;
	}
};
struct PlainCircle : public Intersectable {
	vec3 normal;
	float radius;
	bool rotates;

	PlainCircle(vec3 cen, vec3 n, Material* mat, float rad, bool rot) {
		center = cen; normal = n; material = mat; radius = rad; rotates = rot;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 s = ray.start - center;
		vec3 d = ray.dir;
		hit.t = dot(center - s, normal) / dot(d, normal);
		hit.normal = normal;
		hit.position = s + d * hit.t;
		if (length(vec2(center.x - hit.position.x, center.z - hit.position.z)) > radius) {
			Hit temp;
			return temp;
		}
		hit.material = material;
		return hit;
	}
};

float getHitT(float t1, float t2) {
	if (t2 < 0.0)
		return t1;
	else if (t1 < 0.0)
		return t2;
	else {
		if (t1 < t2)
			return t1;
		else
			return t2;
	}
}

struct Cylinder : public Intersectable {
	float radius;
	float height;
	int layer = 0;


	Cylinder(vec3 cen, float rad, float h, Material* mat, int _layer) {
		center = cen; //bottom center
		radius = rad; height = h; material = mat; layer = _layer;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec4 q1 = quat(normalize(firRD), myT);
		vec4 q2 = quat(normalize(secRD), myT);
		vec3 s = ray.start, d = ray.dir, nC = center;
		if (layer > 0) {
			s = quatRot(q1, ray.start - rotOrigin);
			d = quatRot(q1, ray.dir);
			nC = center - rotOrigin;
			if (layer > 1) {
				s = quatRot(q2, s - nC);
				d = quatRot(q2, d);
				nC = vec3(0, 0, 0);
			}
		}
		vec2 oc = vec2(s.x - nC.x, s.z - nC.z);
		float a = dot(vec2(d.x, d.z), vec2(d.x, d.z));
		float b = 2 * dot(oc, vec2(d.x, d.z));
		float c = dot(oc, oc) - radius * radius;
		float disc = b * b - 4 * a * c;
		if (disc < 0.0) return hit;
		float t1 = (-b - sqrtf(disc)) / (2.0f * a);
		float t2 = (-b + sqrtf(disc)) / (2.0f * a);
		vec3 hitPos1 = s + d * t1;
		vec3 hitPos2 = s + d * t2;
		if (hitPos1.y < nC.y || hitPos1.y > height + nC.y) t1 = -1;
		if (hitPos2.y < nC.y || hitPos2.y > height + nC.y) t2 = -1;
		if (t1 < 0.0 && t2 < 0.0) return hit;
		hit.t = getHitT(t1, t2);
		hit.position = s + d * hit.t;
		hit.material = material;
		hit.normal = normalize(vec3(hit.position.x - nC.x, 0, hit.position.z - nC.z));

		if (dot(hit.normal, d) > 0.0) hit.normal = hit.normal * -1;
		if (layer > 0) {
			if (layer > 1)
				hit.normal = quatRot(quatInv(q2), hit.normal);
			hit.normal = quatRot(quatInv(q1), hit.normal);
		}
		return hit;
	}
};
struct PointLight {
	vec3 location;
	vec3 power;

	PointLight(vec3 _location, vec3 _power) {
		location = _location;
		power = _power;
	}
	double distanceOf(vec3 point) {
		return length((location - point));
	}
	vec3 directionOf(vec3 point) {
		return normalize((location - point));
	}
	vec3 radianceAt(vec3 point) {
		double distance2 = dot(location - point, location - point);
		if (distance2 < epsilon) distance2 = epsilon;
		return power / distance2; // / 4 / M_PI;
	}
};

struct Paraboloid : public Intersectable {
	float curve;
	float height;
	Sphere* mid;
	Sphere* top;
	PointLight* lampLight;
	vec3 focalPoint;

	Paraboloid(vec3 cen, float cur, float h, Material* mat, Sphere* m = nullptr, Sphere* t = nullptr, PointLight* ll = nullptr) {
		center = cen; curve = cur * cur; height = h; material = mat;
		mid = m;
		top = t;
		lampLight = ll;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec4 q1 = quat(normalize(firRD), myT);
		vec4 q2 = quat(normalize(secRD), myT);
		vec4 q3 = quat(normalize(thiRD), myT);
		
		/* A kvaterniós forgatások alapját a Teams-es konzultációból szereztem (Csala Bálint). */
		vec3 s	= quatRot(q3, quatRot(q2, quatRot(q1, ray.start - rotOrigin) - mid->center + rotOrigin) - top->center + mid->center);
		vec3 d	= quatRot(q3, quatRot(q2, quatRot(q1, ray.dir)));
		vec3 nC = center - top->center;
		
		float a = d.x * d.x + d.z * d.z;
		float b = 2 * (s.x * d.x + s.z * d.z) - 2 * (nC.x * d.x - nC.z * d.z) - d.y * curve;
		float c = (s.x - nC.x) * (s.x - nC.x) + (s.z - nC.z) * (s.z - nC.z) - (s.y - nC.y) * curve;
		float dis = b * b - 4 * a * c;
		if (dis < 0) return hit;
		float t1 = (-b - sqrtf(dis)) / (2.0f * a);
		float t2 = (-b + sqrtf(dis)) / (2.0f * a);
		vec3 hitPos1 = s + d * t1;
		vec3 hitPos2 = s + d * t2;

		if (hitPos1.y > height + center.y) t1 = -1;
		if (hitPos2.y > height + center.y) t2 = -1;

		if (t1 < 0.0 && t2 < 0.0) return hit;
		hit.t = getHitT(t1, t2);
		hit.position = ray.start + ray.dir * hit.t;
		hit.material = material;

		hit.normal = normalize(vec3((s.x + d.x * hit.t - center.x) * 2 / curve, -1, (s.z + d.z * hit.t - center.z) * 2 / curve));
		hit.normal = quatRot(quatInv(q1), quatRot(quatInv(q2), quatRot(quatInv(q3), hit.normal))); //Konzultáció alapján
		calculateFocalPoint(q1, q2, q3);
		return hit;
	}
	void calculateFocalPoint(vec4 q1, vec4 q2, vec4 q3) { 
		focalPoint = center + vec3(0, 0.2, 0);

		focalPoint = quatRot(quatInv(q3), focalPoint - top->center);
		focalPoint = focalPoint + top->center;

		focalPoint = quatRot(quatInv(q2), focalPoint - mid->center);
		focalPoint = focalPoint + mid->center;

		focalPoint = quatRot(quatInv(q1), focalPoint - rotOrigin);
		focalPoint = focalPoint + rotOrigin;

		lampLight->location = focalPoint;
	}
	
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye; lookat = _lookat; fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}
};

float rnd() { return (float)rand() / RAND_MAX; }


class Scene {
	std::vector<PointLight*> lights;
	Camera camera;
	vec3 La;

public:
	std::vector<Intersectable*> objects;
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 ks(2, 2, 2);
		Material* parMat  = new Material(vec3(0.20, 0.10, 0.10), ks, 50);
		Material* deskMat = new Material(vec3(0.30, 0.20, 0.10), ks, 50);
		Material* sphMat  = new Material(vec3(0.15, 0.10, 0.05), ks, 50);
		Material* cylMat  = new Material(vec3(0.60, 0.43, 0.15), ks, 50);
		Material* black   = new Material(vec3(0.0f, 0.0f, 0.0f), ks, 50);
		float h = 0.31;
		float r = 0.07;

		//Statikus elemek
		objects.push_back(new Plain		 (vec3(0,   -0.2, 0), normalize(vec3(0.0, 1, 0.0)), deskMat));				//SÍK
		objects.push_back(new Cylinder	 (vec3(0,   -0.4, 0), 0.24f, 0.05f, black, 0));								//Tartó	
		objects.push_back(new PlainCircle(vec3(0, -0.175, 0), normalize(vec3(0.0, 1, 0.0)), cylMat, 0.24f, false));	//Fedõ
		objects.push_back(new Sphere(	  vec3(0, -0.325, 0), r, sphMat, 0));									//1. gömb
		//Mozgó elemek
		PointLight* lampLight = new PointLight(vec3(0,1000,0), vec3(2, 2, 2));										//Lámpa pont fénye

		Sphere* sph1    = new Sphere(	 vec3(0,-0.325 + h      ,0), r   ,  sphMat, 1);								//2. gömb
		Sphere* sph2	= new Sphere(	 vec3(0,-0.325 + 2 * h  ,0), r   ,	sphMat, 2, sph1);						//3. gömb
		Cylinder* cyl1 = new Cylinder( vec3(0,-0.325	        ,0), 0.03, h,cylMat, 1);							//1. rúd
		Cylinder* cyl2 = new Cylinder( vec3(0,-0.325 + h		,0), 0.03, h,cylMat, 2);							//2. rúd
		Paraboloid* par = new Paraboloid(vec3(0,-0.325 + 2*h + r,0), 0.4 , 0.03,parMat, sph1, sph2, lampLight);		//Búra
		objects.push_back(cyl1);
		objects.push_back(cyl2);
		objects.push_back(sph1);
		objects.push_back(sph2);
		objects.push_back(par);
		lights.push_back(lampLight);
	}

	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}

		printf("Rendering time %d milliseconds\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		vec3 newCenter;
		for (Intersectable* ob : objects) {
			Hit hit = ob->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray, vec3 p) {	// for directional lights
		for (Intersectable* ob : objects) {
			Hit o = ob->intersect(ray);
			if (o.t > 0 && (length(p - ray.start) > length(ray.start - o.position))) 
				return true;
		}
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (PointLight* light : lights) {
			vec3 dir = normalize(light->location - hit.position);
			Ray shadowRay(hit.position + hit.normal * epsilon, dir);
			float cosTheta = dot(hit.normal, dir);
			if (cosTheta > 0 && !shadowIntersect(shadowRay, light->location)) {	// shadow computation
				outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + dir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->radianceAt(hit.position) * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}

	void Animate(float dt) {
		camera.Animate(dt);
	}
};

Scene scene;
GPUProgram gpuProgram; // vertex and fragment shaders

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 cvp;
	out vec2 texcoord;

	void main() {
		texcoord = (cvp + vec2(1,1))/2;
		gl_Position = vec4(cvp.x, cvp.y, 0, 1);
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao = 0;	// vertex array object id and texture id
	unsigned int textureID = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void loadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureID);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->loadTexture(image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {

}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space

}
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space

}
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	myT += M_PI / 10;
	scene.Animate(0.05f);
	glutPostRedisplay();
}
