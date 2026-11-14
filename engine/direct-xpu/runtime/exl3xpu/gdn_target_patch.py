"""Semantic transformation of the reviewed 0.33 metadata builder, not a regex."""
import ast
from collections import Counter

EXPECTED_SITES = {'block_table_tensor': 2, 'query_lens': 2,
                  'num_accepted_tokens': 1, 'has_initial_state': 1}


def selector_name(mask):
    if isinstance(mask, ast.Name):
        return {'spec_sequence_masks_cpu': '_exl3_spec_rows',
                'non_spec_sequence_masks_cpu': '_exl3_non_spec_rows'}.get(mask.id)
    if (isinstance(mask, ast.UnaryOp) and isinstance(mask.op, ast.Invert)
            and isinstance(mask.operand, ast.Name)
            and mask.operand.id == 'spec_sequence_masks_cpu'):
        return '_exl3_non_spec_rows'
    return None


class DeviceMaskSites(ast.NodeTransformer):
    def __init__(self):
        self.sites = Counter()

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id not in EXPECTED_SITES:
            return node
        selectors = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        name = selector_name(selectors[0])
        if name is None:
            return node
        self.sites[node.value.id] += 1
        selected = ast.Call(func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()),
                                              attr='select', ctx=ast.Load()),
                            args=[node.value], keywords=[])
        if len(selectors) > 1:
            selected = ast.Subscript(value=selected,
                                     slice=ast.Tuple(elts=[ast.Slice(), *selectors[1:]], ctx=ast.Load()),
                                     ctx=node.ctx)
        return ast.copy_location(selected, node)


def transformed_build(source):
    tree = ast.parse(source)
    patch = DeviceMaskSites()
    tree = patch.visit(tree)
    if dict(patch.sites) != EXPECTED_SITES:
        raise ValueError(f'Unknown GDN device mask sites: {dict(patch.sites)}')
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == 'spec_sequence_masks_cpu'
                   and isinstance(n.value, ast.Compare)]
    if len(assignments) != 1:
        raise ValueError('Unknown GDN speculative CPU mask construction')
    anchor = assignments[0]
    injected = ast.parse('_exl3_spec_rows = _Exl3HostRows(spec_sequence_masks_cpu)\n'
                         '_exl3_non_spec_rows = _Exl3HostRows(~spec_sequence_masks_cpu)').body
    for parent in ast.walk(tree):
        for _, field in ast.iter_fields(parent):
            if isinstance(field, list) and anchor in field:
                pos = field.index(anchor) + 1
                field[pos:pos] = injected
    ast.fix_missing_locations(tree)
    # Reject any remaining mask access on the explicitly identified device tensors.
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in EXPECTED_SITES:
            selectors = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if selector_name(selectors[0]):
                raise ValueError('Unaccounted GDN device/CPU mask access')
    return ast.unparse(tree), dict(patch.sites)
