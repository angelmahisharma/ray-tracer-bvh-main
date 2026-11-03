#include "rt/bvh.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace rt;

namespace {
int failures = 0;
void check(bool ok, const char* name) {
    if (!ok) { std::cerr << "FAIL: " << name << '\n'; ++failures; }
}
bool near(float a, float b, float eps=1e-4f) { return std::fabs(a-b) <= eps; }

Triangle triangle(float x, float y, float z) {
    Triangle t;
    t.v0=Vec3(x-0.4f,y-0.4f,z); t.v1=Vec3(x+0.4f,y-0.4f,z); t.v2=Vec3(x,y+0.4f,z);
    t.n0=t.n1=t.n2=Vec3(0,0,1);
    return t;
}

bool intersect_triangle(const Triangle& t, const Ray& ray, float& best) {
    Vec3 e1=t.v1-t.v0, e2=t.v2-t.v0, p=cross(ray.dir,e2);
    float det=dot(e1,p); if(std::fabs(det)<1e-8f)return false;
    float inv=1.f/det; Vec3 q=ray.origin-t.v0; float u=dot(q,p)*inv;
    if(u<0.f||u>1.f)return false;
    Vec3 r=cross(q,e1); float v=dot(ray.dir,r)*inv;
    if(v<0.f||u+v>1.f)return false;
    float tval=dot(e2,r)*inv;
    if(tval<=ray.tmin||tval>=best)return false;
    best=tval; return true;
}

struct Result { bool hit=false; float t=kBigFloat; uint32_t id=0; };
Result brute(const Scene& s,const Ray& ray) {
    Result out; out.t=ray.tmax;
    for(uint32_t i=0;i<s.triangles.size();++i) if(intersect_triangle(s.triangles[i],ray,out.t)){out.hit=true;out.id=i;}
    return out;
}
Result accelerated(const Scene& s,const BVH& bvh,const Ray& ray) {
    Result out; out.t=ray.tmax;
    auto test=[&](uint32_t id,float& max_t){return intersect_triangle(s.triangles[id],ray,max_t);};
    out.hit=bvh.traverse(ray,out.t,test,out.id);
    return out;
}
Scene grid_scene(int side) {
    Scene s;
    for(int z=0;z<3;++z) for(int y=0;y<side;++y) for(int x=0;x<side;++x)
        s.triangles.push_back(triangle((x-side/2)*1.1f,(y-side/2)*1.1f,-float(z)*1.25f));
    return s;
}
void compare_scene(const Scene& s,const char* label) {
    BVH median,sah; median.build(s,BVHStrategy::Median); sah.build(s,BVHStrategy::SAH);
    bool valid=median.stats.references==s.triangles.size()&&sah.stats.references==s.triangles.size()
        &&median.stats.nodes==median.stats.internal+median.stats.leaves
        &&sah.stats.nodes==sah.stats.internal+sah.stats.leaves
        &&median.primitives.size()==s.triangles.size()&&sah.primitives.size()==s.triangles.size();
    check(valid,label);
    for(int y=-7;y<=7;++y) for(int x=-7;x<=7;++x) {
        Ray ray(Vec3(x*.31f+.037f,y*.29f+.061f,3.f),normalize(Vec3(.013f,-.009f,-1.f)));
        Result ref=brute(s,ray), m=accelerated(s,median,ray), a=accelerated(s,sah,ray);
        check(ref.hit==m.hit&&ref.hit==a.hit,"closest-hit presence agrees with brute force");
        if(ref.hit) check(near(ref.t,m.t,2e-4f)&&near(ref.t,a.t,2e-4f),"closest-hit distance agrees with brute force");
    }
}
}

int main() {
    std::cout << "AABB slab tests\n";
    AABB box; check(!box.valid(),"empty AABB is invalid");
    box.expand(Vec3(-1,-1,-1)); box.expand(Vec3(1,1,1)); float entry=-1.f;
    check(intersect_aabb(box,Ray(Vec3(0,0,3),Vec3(0,0,-1)),kBigFloat,entry)&&near(entry,2.f),"axis-aligned ray hits box");
    check(intersect_aabb(box,Ray(Vec3(0,0,0),Vec3(1,0,0)),kBigFloat,entry)&&near(entry,kRayEps),"ray starting inside box");
    check(!intersect_aabb(box,Ray(Vec3(2,0,3),Vec3(0,0,-1)),kBigFloat,entry),"parallel ray outside slab misses");
    check(intersect_aabb(box,Ray(Vec3(1,1,3),Vec3(0,0,-1)),kBigFloat,entry),"ray on box edge hits");
    check(intersect_aabb(box,Ray(Vec3(0,0,3),Vec3(0,0,-1),kRayEps,1.f),kBigFloat,entry)==false,"tmax clips box hit");
    check(!intersect_aabb(AABB{},Ray(Vec3(0,0,3),Vec3(0,0,-1)),kBigFloat,entry),"invalid box misses");

    std::cout << "Empty and one-triangle trees\n";
    Scene empty; BVH empty_m,empty_s; empty_m.build(empty,BVHStrategy::Median); empty_s.build(empty,BVHStrategy::SAH);
    check(empty_m.nodes.empty()&&empty_s.nodes.empty(),"empty scene creates empty BVHs");
    check(!accelerated(empty,empty_m,Ray(Vec3(0,0,1),Vec3(0,0,-1))).hit,"empty BVH traversal misses");
    Scene one; one.triangles.push_back(triangle(0,0,0));
    BVH one_m,one_s; one_m.build(one,BVHStrategy::Median); one_s.build(one,BVHStrategy::SAH);
    check(one_m.nodes.size()==1&&one_m.stats.leaves==1&&one_m.stats.internal==0,"single-triangle median is one leaf");
    check(one_s.nodes.size()==1&&one_s.stats.leaves==1&&one_s.stats.references==1,"single-triangle SAH is one leaf");
    Ray center(Vec3(0,0,2),Vec3(0,0,-1));
    check(brute(one,center).hit&&accelerated(one,one_m,center).hit&&accelerated(one,one_s,center).hit,"single-triangle hit in all modes");

    std::cout << "Median and SAH correctness comparisons\n";
    compare_scene(grid_scene(1),"one column build references and counts");
    compare_scene(grid_scene(2),"small scene build references and counts");
    compare_scene(grid_scene(9),"multi-leaf scene build references and counts");
    Scene same_centroids; for(int i=0;i<37;++i) same_centroids.triangles.push_back(triangle(0,0,-float(i)*.2f));
    compare_scene(same_centroids,"repeated centroid scene terminates and matches brute force");
    Scene large_leaf=grid_scene(3); BVH leaf_bvh; leaf_bvh.build(large_leaf,BVHStrategy::Median,2);
    check(leaf_bvh.stats.max_leaf<=2,"configurable leaf size is respected");

    if(failures){std::cerr<<failures<<" BVH check(s) failed\n";return 1;}
    std::cout<<"All BVH checks passed\n"; return 0;
}
