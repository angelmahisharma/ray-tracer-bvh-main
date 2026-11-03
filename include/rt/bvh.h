#pragma once
#include "scene.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

namespace rt {

inline bool intersect_aabb(const AABB& b, const Ray& r, float limit, float& entry) {
    if (!b.valid()) return false;
    float lo = r.tmin, hi = std::min(r.tmax, limit);
    for (int a=0; a<3; ++a) {
        const float o=r.origin[a], d=r.dir[a], mn=b.lo[a], mx=b.hi[a];
        if (d == 0.f) { if (o < mn || o > mx) return false; continue; }
        float x=(mn-o)/d, y=(mx-o)/d; if (x>y) std::swap(x,y);
        lo=std::max(lo,x); hi=std::min(hi,y); if (lo>hi) return false;
    }
    entry=lo; return hi>=r.tmin;
}

enum class BVHStrategy { Median, SAH };
struct FlatBVHNode { AABB bounds; uint32_t first=0, count=0, right=0; bool leaf() const { return count!=0; } };
struct BVHStats { size_t nodes=0, internal=0, leaves=0, references=0; uint32_t max_depth=0, max_leaf=0; double leaf_sum=0; double average_triangles_per_leaf() const { return leaves?leaf_sum/leaves:0.; } };
struct BVH {
    std::vector<FlatBVHNode> nodes;
    std::vector<uint32_t> primitives;
    BVHStats stats;
    void build(const Scene& scene, BVHStrategy strategy, uint32_t leaf_size=4) {
        nodes.clear(); primitives.resize(scene.triangles.size()); std::iota(primitives.begin(),primitives.end(),0u); stats={};
        if (primitives.empty()) return;
        std::vector<uint32_t> order=primitives;
        struct Temp { AABB box; int left=-1,right=-1; uint32_t first=0,count=0; };
        std::vector<Temp> temp;
        auto bounds=[&](size_t a,size_t z){ AABB b; for(size_t i=a;i<z;++i)b.expand(scene.triangles[order[i]].bounds()); return b; };
        auto build_rec=[&](auto&& self,size_t a,size_t z,uint32_t depth)->int {
            int n=(int)temp.size(); temp.emplace_back(); temp[n].box=bounds(a,z); stats.max_depth=std::max(stats.max_depth,depth);
            size_t count=z-a;
            auto leaf=[&](){ temp[n].first=(uint32_t)a; temp[n].count=(uint32_t)count; stats.leaves++; stats.references+=count; stats.leaf_sum+=count; stats.max_leaf=std::max(stats.max_leaf,(uint32_t)count); return n; };
            if(count<=std::max(1u,leaf_size)) return leaf();
            AABB cb; for(size_t i=a;i<z;++i)cb.expand(scene.triangles[order[i]].centroid());
            int axis=cb.longest_axis(); size_t mid=a+count/2; bool split=false;
            if(strategy==BVHStrategy::SAH && cb.extent()[axis]>0.f && temp[n].box.surface_area()>1e-30f) {
                constexpr int B=16; struct Bin{AABB b;uint32_t c=0;}; std::array<Bin,B> bins{};
                float mn=cb.lo[axis], span=cb.extent()[axis];
                for(size_t i=a;i<z;++i){int q=std::min(B-1,(int)((scene.triangles[order[i]].centroid()[axis]-mn)/span*B));bins[q].c++;bins[q].b.expand(scene.triangles[order[i]].bounds());}
                float best=(float)count; int bestcut=-1; AABB lb[B],rb[B]; uint32_t lc[B]{},rc[B]{}; AABB acc;uint32_t c=0;
                for(int i=0;i<B;++i){acc.expand(bins[i].b);c+=bins[i].c;lb[i]=acc;lc[i]=c;} acc=AABB{};c=0;
                for(int i=B-1;i>=0;--i){acc.expand(bins[i].b);c+=bins[i].c;rb[i]=acc;rc[i]=c;}
                for(int i=0;i<B-1;++i)if(lc[i]&&rc[i+1]){float cost=1.f+(lb[i].surface_area()*lc[i]+rb[i+1].surface_area()*rc[i+1])/temp[n].box.surface_area();if(cost<best){best=cost;bestcut=i;}}
                if(bestcut>=0){auto it=std::partition(order.begin()+a,order.begin()+z,[&](uint32_t id){int q=std::min(B-1,(int)((scene.triangles[id].centroid()[axis]-mn)/span*B));return q<=bestcut;});mid=(size_t)(it-order.begin());split=mid>a&&mid<z;}
            }
            if(!split){std::stable_sort(order.begin()+a,order.begin()+z,[&](uint32_t x,uint32_t y){return scene.triangles[x].centroid()[axis]<scene.triangles[y].centroid()[axis];}); mid=a+count/2; split=true;}
            if(!split||mid<=a||mid>=z)return leaf();
            // Recursive calls may reallocate temp; do not retain references to
            // its elements while descending.
            const int left=self(self,a,mid,depth+1);
            const int right=self(self,mid,z,depth+1);
            temp[n].left=left; temp[n].right=right; return n;
        };
        build_rec(build_rec,0,order.size(),0);
        primitives.clear();
        auto flatten=[&](auto&& self,int t)->uint32_t {uint32_t i=(uint32_t)nodes.size();nodes.push_back({});nodes[i].bounds=temp[t].box;
            if(temp[t].count){nodes[i].first=(uint32_t)primitives.size();nodes[i].count=temp[t].count;for(uint32_t k=0;k<temp[t].count;++k)primitives.push_back(order[temp[t].first+k]);}
            else{++stats.internal;self(self,temp[t].left);nodes[i].right=(uint32_t)nodes.size();self(self,temp[t].right);}return i;};
        flatten(flatten,0); stats.nodes=nodes.size();
    }
    template<class Intersector> bool traverse(const Ray& ray, float& closest, Intersector&& intersect, uint32_t& tri) const {
        if(nodes.empty())return false; bool found=false; std::vector<uint32_t> stack{0};
        while(!stack.empty()){uint32_t i=stack.back();stack.pop_back();const auto& n=nodes[i];float en;
            if(!intersect_aabb(n.bounds,ray,closest,en))continue;
            if(n.leaf()){for(uint32_t k=0;k<n.count;++k){uint32_t id=primitives[n.first+k];if(intersect(id,closest)){tri=id;found=true;}}}
            else{float le,re;bool hl=intersect_aabb(nodes[i+1].bounds,ray,closest,le),hr=intersect_aabb(nodes[n.right].bounds,ray,closest,re);if(hl&&hr){if(le<re){stack.push_back(n.right);stack.push_back(i+1);}else{stack.push_back(i+1);stack.push_back(n.right);}}else if(hl)stack.push_back(i+1);else if(hr)stack.push_back(n.right);}
        } return found;
    }
};
}
