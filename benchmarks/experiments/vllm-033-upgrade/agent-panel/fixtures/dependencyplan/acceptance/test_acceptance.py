import copy,unittest
from buildplan import plan_targets,UnknownTarget,DependencyCycle
class Acceptance(unittest.TestCase):
 def test_ready_lexicographic(self):
  graph={'z':[],'a':['z'],'b':[],'c':['b'],'x':['a','c']}
  self.assertEqual(plan_targets(graph,['x']),['b','c','z','a','x'])
 def test_reachable_only(self):self.assertEqual(plan_targets({'app':['lib'],'lib':[],'bad':['absent'],'loop':['loop']},['app']),['lib','app'])
 def test_unknown_target(self):
  with self.assertRaises(UnknownTarget) as error:plan_targets({'a':[]},['missing'])
  self.assertEqual(error.exception.name,'missing')
 def test_unknown_dependency(self):
  with self.assertRaises(UnknownTarget) as error:plan_targets({'a':['missing']},['a'])
  self.assertEqual(error.exception.name,'missing')
 def test_cycle_and_downstream(self):
  with self.assertRaises(DependencyCycle) as error:plan_targets({'a':['b'],'b':['a'],'consumer':['a'],'okay':[]},['consumer','okay'])
  self.assertEqual(error.exception.blocked,['a','b','consumer'])
 def test_self_cycle(self):
  with self.assertRaises(DependencyCycle) as error:plan_targets({'a':['a']},['a'])
  self.assertEqual(error.exception.blocked,['a'])
 def test_duplicates_and_no_mutation(self):
  graph={'z':[],'a':['z','z'],'b':[]};targets=['a','b','a'];snapshot=copy.deepcopy((graph,targets))
  self.assertEqual(plan_targets(graph,targets),['b','z','a']);self.assertEqual((graph,targets),snapshot)
 def test_empty(self):self.assertEqual(plan_targets({'bad':['missing']},[]),[])
 def test_deep_chain(self):
  graph={f'n{i:04}':[f'n{i-1:04}'] if i else [] for i in range(2100)}
  self.assertEqual(plan_targets(graph,['n2099']),[f'n{i:04}' for i in range(2100)])
