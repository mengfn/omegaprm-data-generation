import json
import re
import unittest
from tools.visualize_problem import prepare_graph, render_html


def fixture():
    return {'states': [{'id': i, 'prefix': p, 'mc': .5} for i, p in enumerate(['', 'a', 'ab', 'abc', 'probe'])],
            'edges': [{'parent_id': a, 'child_id': b, 'action': t} for a,b,t in [(0,1,'a'),(0,2,'ab'),(1,3,'bc'),(2,3,'c')]]}


class Visualization(unittest.TestCase):
    def test_shared_child_and_disconnected_probe(self):
        g = prepare_graph(fixture())
        self.assertEqual(len(g['edges']), 4)
        self.assertEqual(g['probe_count'], 1)
        self.assertGreater(g['nodes'][3]['y'], g['nodes'][2]['y'])
        self.assertTrue(g['nodes'][4]['probe'])

    def test_root_only(self):
        g = prepare_graph({'states':[{'id':0,'prefix':'','mc':1}], 'edges':[]})
        self.assertEqual(g['probe_count'], 0)
        self.assertFalse(g['nodes'][0]['probe'])

    def test_model_text_cannot_escape_script(self):
        data = fixture()
        data['question'] = '</script><script>alert(1)</script>& 中文'
        html = render_html(data)
        payload = re.search(r'type="application/json">(.*?)</script>', html, re.S).group(1)
        self.assertNotIn('<', payload)
        self.assertEqual(json.loads(payload)['question'], data['question'])

    def test_invalid_edges(self):
        for update in [{'child_id':99}, {'action':'wrong'}]:
            with self.subTest(update=update):
                data = fixture()
                data['edges'][0].update(update)
                with self.assertRaises(ValueError): prepare_graph(data)

    def test_invalid_ids_and_mc(self):
        for update in [{'id':1}, {'mc':float('nan')}, {'mc':2}]:
            with self.subTest(update=update):
                data = fixture()
                data['states'][0].update(update)
                with self.assertRaises(ValueError): prepare_graph(data)

    def test_wrong_input(self):
        with self.assertRaises(ValueError): prepare_graph({'samples':[]})
