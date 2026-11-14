#include "bounded_partition_cache.h"
#include <cassert>
#include <memory>
#include <iostream>

struct State { bool busy; bool unsafe_destroy=false; unsigned waits=0; };
struct Value {
    std::shared_ptr<State> state;
    explicit Value(std::shared_ptr<State> s):state(std::move(s)) {}
    Value(Value&&)=default;
    ~Value() { if (state && state->busy) state->unsafe_destroy=true; }
};
int main() {
    auto ready=[](const Value& v){return !v.state->busy;};
    auto wait=[](const Value& v){v.state->busy=false;++v.state->waits;};
    {
        auto a=std::make_shared<State>(State{true}),b=std::make_shared<State>(State{true}),c=std::make_shared<State>(State{false});
        BoundedPartitionCache<int,Value> cache(2);
        cache.get(1,[&]{return Value(a);},ready,wait);
        cache.get(2,[&]{return Value(b);},ready,wait);
        cache.get(1,[]{throw std::runtime_error("hot hit compiled");return Value(nullptr);},ready,wait);
        cache.get(3,[&]{return Value(c);},ready,wait);
        assert(cache.size()==2 && cache.stats.peak_entries==2);
        assert(cache.stats.hits==1 && cache.stats.misses==3 && cache.stats.evictions==1);
        assert(cache.stats.pressure_waits==1 && b->waits==1 && !b->unsafe_destroy);
        a->busy=false;
    }
    {
        auto a=std::make_shared<State>(State{true}),b=std::make_shared<State>(State{false}),c=std::make_shared<State>(State{false});
        BoundedPartitionCache<int,Value> cache(2);
        cache.get(1,[&]{return Value(a);},ready,wait);
        cache.get(2,[&]{return Value(b);},ready,wait);
        cache.get(3,[&]{return Value(c);},ready,wait);
        assert(cache.stats.pressure_waits==0 && !b->unsafe_destroy && a->waits==0);
        bool failed=false;
        try {cache.get(4,[]{throw std::runtime_error("compile fails");return Value(nullptr);},ready,wait);}
        catch (const std::runtime_error&) {failed=true;}
        assert(failed && cache.size()<2 && !a->unsafe_destroy);
        a->busy=false;
    }
    std::cout<<"PASS bounded LRU, hot hits, async retirement, pressure-only wait, compile failure\n";
}
