class UnknownTarget(ValueError):
    def __init__(self,name):
        self.name=name
        super().__init__(name)

class DependencyCycle(ValueError):
    def __init__(self,blocked):
        self.blocked=sorted(blocked)
        super().__init__(', '.join(self.blocked))

def plan_targets(graph,targets):
    result=[]
    seen=set()
    def visit(name):
        if name in seen:return
        seen.add(name)
        for dependency in sorted(graph[name]):visit(dependency)
        result.append(name)
    for target in sorted(set(targets)):visit(target)
    return result
