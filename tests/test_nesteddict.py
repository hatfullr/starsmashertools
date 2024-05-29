import starsmashertools.helpers.nesteddict
import unittest
import basetest
import copy
import os

class TestNestedDict(basetest.BaseTest):
    def tearDown(self):
        if os.path.isfile('test.pickle'): os.remove('test.pickle')
        
    def test_getitem(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        
        self.assertEqual(d['2'], None)
        self.assertEqual(d['1'], {'1a':{'1ai':1,'1aii':2},'1b':{'1bi':3}})
        self.assertEqual(d['1','1a'], {'1ai':1,'1aii':2})
        self.assertEqual(d['1','1b'], {'1bi':3})
        self.assertEqual(d['1','1a','1ai'], 1)
        self.assertEqual(d['1','1a','1aii'], 2)
        self.assertEqual(d['1','1b','1bi'], 3)

        d = starsmashertools.helpers.nesteddict.NestedDict({'1':{'1a':1}})
        self.assertEqual(d['1'], {'1a':1})
        self.assertEqual(d['1','1a'], 1)

    def test_setitem(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({})
        d['1'] = {'1a' : 1}
        self.assertEqual(d['1','1a'], 1)
        d['1','1a'] = 2
        self.assertEqual(d['1','1a'], 2)
        self.assertEqual(len(d), 2)
        d['1','1b'] = 4
        self.assertEqual(d['1','1b'], 4)
        self.assertEqual(len(d), 3)
        self.assertEqual(len(d['1']), 2)
        d['1','1a'] = {'1ai' : None}
        self.assertEqual(d['1','1a','1ai'], None)
        d['2'] = 'hello'
        self.assertEqual(d['2'], 'hello')

    def test_keys(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        expected = ['1', ('1', '1a'), ('1', '1a', '1ai'), ('1', '1a', '1aii'), ('1', '1b'), ('1', '1b', '1bi'), '2']
        found = d.keys()
        self.assertEqual(len(expected), len(found), msg = 'found = %s, expected = %s' % (found, expected))
        for key1, key2 in zip(expected, found):
            self.assertEqual(key1, key2)
            self.assertIn(key1, found)

        expected = [('1', '1a'), ('1', '1a', '1ai'), ('1', '1a', '1aii')]
        found = d.keys(stems = [('1', '1a')])
        self.assertEqual(len(expected), len(found))
        for key1, key2 in zip(expected, found):
            self.assertEqual(key1, key2)
            self.assertIn(key1, found)

        expected = ['2']
        found = d.keys(stems = ['2'])
        self.assertEqual(1, len(found))
        for key1, key2 in zip(expected, found):
            self.assertEqual(key1, key2)
            self.assertIn(key1, found)

    def test_values(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        expected = [
            {'1a': {'1ai': 1, '1aii': 2}, '1b': {'1bi': 3}}, # d['1']
            {'1ai': 1, '1aii': 2}, # d['1','1a']
            1, # d['1','1a','1ai']
            2, # d['1','1a','1aii']
            {'1bi': 3}, # d['1','1b']
            3, # d['1','1b','1bi']
            None, # d['2']
        ]
        found = d.values()
        self.assertEqual(len(expected), len(found))
        for key, val1, val2 in zip(d.keys(), expected, found):
            self.assertEqual(val1, val2, msg = 'key = %s: expected = %s, found = %s' % (key, val1, val2))
            self.assertIn(val1, found)

        
        expected = [
            {'1ai': 1, '1aii': 2}, # d['1','1a']
            1, # d['1','1a','1ai']
            2, # d['1','1a','1aii']
        ]
        found = d.values(stems = [('1', '1a')])
        self.assertEqual(len(expected), len(found))
        for key, val1, val2 in zip(d.keys(), expected, found):
            self.assertEqual(val1, val2, msg = 'key = %s: expected = %s, found = %s' % (key, val1, val2))
            self.assertIn(val1, found)

        expected = [None]
        found = d.values(stems = ['2'])
        self.assertEqual(1, len(found))
        for key, val1, val2 in zip(d.keys(), expected, found):
            self.assertEqual(val1, val2, msg = 'key = %s: expected = %s, found = %s' % (key, val1, val2))
            self.assertIn(val1, found)


    def test_items(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        expected = [
            ('1', {'1a': {'1ai': 1, '1aii': 2}, '1b': {'1bi': 3}}), # d['1']
            (('1','1a'), {'1ai': 1, '1aii': 2}), # d['1','1a']
            (('1','1a','1ai'), 1), # d['1','1a','1ai']
            (('1','1a','1aii'),2), # d['1','1a','1aii']
            (('1','1b'),{'1bi': 3}), # d['1','1b']
            (('1','1b','1bi'), 3), # d['1','1b','1bi']
            ('2', None), # d['2']
        ]

        found = d.items()
        self.assertEqual(len(expected), len(found))
        for item1, item2 in zip(expected, found):
            self.assertEqual(item1, item2)
            self.assertIn(item1, found)

        expected = [
            (('1','1a'), {'1ai': 1, '1aii': 2}), # d['1','1a']
            (('1','1a','1ai'), 1), # d['1','1a','1ai']
            (('1','1a','1aii'),2), # d['1','1a','1aii']
        ]
        found = d.items(stems = [('1','1a')])
        self.assertEqual(len(expected), len(found))
        for item1, item2 in zip(expected, found):
            self.assertEqual(item1, item2)
            self.assertIn(item1, found)

        expected = [('2', None)]
        found = d.items(stems = ['2'])
        self.assertEqual(1, len(found))
        for item1, item2 in zip(expected, found):
            self.assertEqual(item1, item2)
            self.assertIn(item1, found)

    def test_branches(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        expected = [('1','1a','1ai'), ('1','1a','1aii'), ('1','1b','1bi'), '2']
        branches = d.branches()
        self.assertEqual(len(expected), len(branches))
        for exp, branch in zip(expected, branches):
            self.assertEqual(exp, branch)
            self.assertIn(exp, branches)

        expected = [('1','1a','1ai'), ('1','1a','1aii')]
        branches = d.branches(stems = [('1', '1a')])
        #for branch in branches: print(branch)
        self.assertEqual(len(expected), len(branches))
        for exp, branch in zip(expected, branches):
            self.assertEqual(exp, branch)
            self.assertIn(exp, branches)

        expected = [('1','1b','1bi')]
        branches = d.branches(stems = [('1', '1b')])
        self.assertEqual(len(expected), len(branches))
        for exp, branch in zip(expected, branches):
            self.assertEqual(exp, branch)
            self.assertIn(exp, branches)

        expected = ['2']
        branches = d.branches(stems = ['2'])
        self.assertEqual(len(expected), len(branches))
        for exp, branch in zip(expected, branches):
            self.assertEqual(exp, branch)
            self.assertIn(exp, branches)
    
    def test_stems(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                    '1aiii' : {
                        '1aiii1' : 1,
                    },
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        expected = ['1', ('1','1a'), ('1','1a','1aiii'), ('1','1b')]
        stems = d.stems()
        self.assertEqual(len(expected), len(stems))
        for exp, stem in zip(expected, stems):
            self.assertEqual(exp, stem)
            self.assertIn(exp, stems)

        expected = [('1', '1a'), ('1', '1a', '1aiii')]
        stems = d.stems(stems = [('1', '1a')])
        self.assertEqual(len(expected), len(stems))
        for exp, stem in zip(expected, stems):
            self.assertEqual(exp, stem)
            self.assertIn(exp, stems)

        self.assertEqual(
            ['1', ('1', '1a'), ('1', '1a', '1aiii')],
            list(d.get_stems(('1','1a','1aiii','1aiii1'))),
        )
        # stems can be branches, branches can't be stems
        stems = d.stems()
        for branch in d.branches():
            self.assertNotIn(branch, stems)
        
    def test_leaves(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        expected = [1, 2, 3, None]
        leaves = d.leaves()
        self.assertEqual(len(expected), len(leaves))
        for exp, leaf in zip(expected, leaves):
            self.assertEqual(exp, leaf)
            self.assertIn(exp, leaves)

        expected = [1, 2]
        leaves = d.leaves(stems = [('1', '1a')])
        self.assertEqual(len(expected), len(leaves))
        for exp, leaf in zip(expected, leaves):
            self.assertEqual(exp, leaf)
            self.assertIn(exp, leaves)

    def test_flowers(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        expected = [
            (('1','1a','1ai'), 1), # d['1','1a','1ai']
            (('1','1a','1aii'),2), # d['1','1a','1aii']
            (('1','1b','1bi'), 3), # d['1','1b','1bi']
            ('2', None), # d['2']
        ]

        found = d.flowers()
        self.assertEqual(len(expected), len(found))
        for item1, item2 in zip(expected, found):
            self.assertEqual(item1, item2)
            self.assertIn(item1, found)


        expected = [
            (('1','1a','1ai'), 1), # d['1','1a','1ai']
            (('1','1a','1aii'),2), # d['1','1a','1aii']
        ]

        found = d.flowers(stems = [('1', '1a')])
        self.assertEqual(len(expected), len(found))
        for item1, item2 in zip(expected, found):
            self.assertEqual(item1, item2)
            self.assertIn(item1, found)
            
    def test_list(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({'1':{'1a':1}})
        expected = [('1','1a')]
        for k1, k2 in zip(expected, list(d)):
            self.assertEqual(k1, k2)

    def test_len(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({'1':{'1a':1}})
        self.assertEqual(len(d), 2)

        d = starsmashertools.helpers.nesteddict.NestedDict({'1':1})
        self.assertEqual(len(d), 1)

    def test_contains(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        self.assertIn('1', d)
        self.assertIn(('1',), d)
        self.assertIn(('1','1a'), d)
        self.assertIn(('1','1a','1ai'), d)
        self.assertIn(('1','1a','1aii'),d)
        self.assertIn(('1','1b'), d)
        self.assertIn(('1','1b','1bi'), d)
        self.assertIn('2', d)
        self.assertIn(('2',), d)
        self.assertNotIn(('2', None), d)
        
        branches = d.branches()
        self.assertIn(('1','1a','1ai'), branches)
        self.assertIn(('1','1a','1aii'),branches)
        self.assertIn(('1','1b','1bi'), branches)
        self.assertIn('2', branches)
        self.assertIn(('2',), branches)
        self.assertNotIn(('2', None), branches)

    def test_iter(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        keys = d.branches()
        for found, expected in zip(d, keys):
            self.assertEqual(expected, found)

    def test_clear(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({'1':{'1a':1}})
        d.clear()
        self.assertEqual(len(d), 0)
        self.assertEqual(len(d.keys()), 0)
        self.assertEqual(len(d.values()), 0)
        self.assertEqual(len(d.items()), 0)
        self.assertEqual(d, {})
        self.assertFalse(d)

        with self.assertRaises(KeyError): d['1']
        with self.assertRaises(KeyError): d['1', '1a']
    
    def test_copy(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({'1':{'1a':1}})
        cpy = copy.copy(d)
        self.assertEqual(d, cpy)
        self.assertIsNot(d, cpy)
        for key, val in d.items():
            self.assertEqual(val, cpy[key])
            self.assertIs(val, cpy[key])

    def test_deepcopy(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({'1':{'1a':[1]}})
        cpy = copy.deepcopy(d)
        self.assertEqual(d, cpy)
        self.assertIsNot(d, cpy)
        for key, val in d.items():
            self.assertEqual(val, cpy[key])
            self.assertIsNot(val, cpy[key])

    def test_fromkeys(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({'1':{'1a':[1]}})
        with self.assertRaises(KeyError): d.fromkeys(['1', ('1','1a')])
        self.assertEqual(d.fromkeys(['1']), {'1':None})
        self.assertEqual(d.fromkeys([('1','1a')])['1','1a'], None)
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : None,
                '1b' : None,
                '1c' : {
                    '1ci' : 1
                },
            },
        })

        self.assertEqual(d.fromkeys([('1','1c','1ci')]), {'1':{'1c':{'1ci':None}}})

    def test_get(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        self.assertEqual(d.get('2'), None)
        self.assertEqual(d.get(('1','1a','1ai')), 1)
        self.assertEqual(d.get('3'), None)

    def test_pop(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        val = d.pop('2')
        self.assertNotIn('2', d)
        self.assertEqual(val, None)

        val = d.pop(('1','1b','1bi'))
        self.assertEqual(val, 3)
        self.assertNotIn(('1','1b','1bi'), d)

    def test_popitem(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        self.assertEqual(d.popitem(), ('2',None))
        self.assertEqual(d.popitem(), (('1','1b','1bi'), 3))
        self.assertEqual(d.popitem(), (('1','1b'), {}))
        self.assertEqual(d.popitem(), (('1','1a','1aii'), 2))
        self.assertEqual(d.popitem(), (('1','1a','1ai'), 1))
        self.assertEqual(d.popitem(), (('1','1a'), {}))
        self.assertEqual(d.popitem(), ('1', {}))
        with self.assertRaises(KeyError): d.popitem()

        d['a','1'] = 1
        d['a','2'] = 2
        d['c','r','a','b'] = None
        self.assertEqual(d.popitem(), (('c','r','a','b'), None))
        self.assertEqual(d.popitem(), (('c','r','a'), {}))
        self.assertEqual(d.popitem(), (('c','r'), {}))
        self.assertEqual(d.popitem(), ('c', {}))
        self.assertEqual(d.popitem(), (('a','2'), 2))
        self.assertEqual(d.popitem(), (('a','1'), 1))
        self.assertEqual(d.popitem(), ('a', {}))
        with self.assertRaises(KeyError): d.popitem()
        
    def test_reversed(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        expected = reversed(['1', ('1', '1a'), ('1', '1a', '1ai'), ('1', '1a', '1aii'), ('1', '1b'), ('1', '1b', '1bi'), '2'])
        for key1, key2 in zip(reversed(d), expected):
            self.assertEqual(key1, key2)
            
    def test_setdefault(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        self.assertEqual(d.setdefault(('1','1a','1aiii'), 66), 66)
        self.assertEqual(d['1','1a','1aiii'], 66)
        self.assertEqual(len(d['1','1a']), 3)

    def test_update(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        other = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 5,
                },
            },
            '2' : 55,
        })

        upd = copy.deepcopy(d)
        upd.update(other)
        self.assertEqual(upd['1','1a','1ai'], 5)
        self.assertEqual(upd['2'], 55)
        self.assertEqual(upd['1'], {'1a':{'1ai':5,'1aii':2},'1b':{'1bi':3}})
        self.assertEqual(upd['1','1a'], {'1ai':5,'1aii':2})
        self.assertEqual(upd['1','1b'], {'1bi':3})
        self.assertEqual(upd['1','1a','1aii'], 2)
        self.assertEqual(upd['1','1b','1bi'], 3)

    def test_comparisons(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({})
        other = starsmashertools.helpers.nesteddict.NestedDict({})
        with self.assertRaises(TypeError): d < other
        with self.assertRaises(TypeError): d > other
        with self.assertRaises(TypeError): d <= other
        with self.assertRaises(TypeError): d >= other

        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        other = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })

        self.assertEqual(d, other)
        other.pop('2')
        self.assertNotEqual(d, other)
        

    def test_del(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        d['1','1a','1ai'] = 2
        del d['1','1a','1ai']
        self.assertNotIn('1ai', d['1','1a'])
        d['1','1a','1ai'] = None
        self.assertEqual(list(d['1','1a'].values())[-1], None)
    
    def test_or_ior(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        other = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1aii' : 200,
                },
            },
            '3' : {
                'yippee' : 'haha',
            },
        })

        combined = d | other
        self.assertEqual(combined['1','1a','1ai'], 1)
        self.assertEqual(combined['1','1a','1aii'], 200)
        self.assertEqual(combined['1','1b','1bi'], 3)
        self.assertEqual(combined['2'], None)
        self.assertEqual(combined['3'], {'yippee':'haha'})

        combined = other | d
        self.assertEqual(combined['1','1a','1ai'], 1)
        self.assertEqual(combined['1','1a','1aii'], 2)
        self.assertEqual(combined['1','1b','1bi'], 3)
        self.assertEqual(combined['2'], None)
        self.assertEqual(combined['3'], {'yippee':'haha'})

        # Testing ior
        d |= other
        self.assertEqual(d['1','1a','1ai'], 1)
        self.assertEqual(d['1','1a','1aii'], 200)
        self.assertEqual(d['1','1b','1bi'], 3)
        self.assertEqual(d['2'], None)
        self.assertEqual(d['3'], {'yippee':'haha'})

    def test_pickle(self):
        import pickle
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
        })
        with open('test.pickle','wb') as f:
            pickle.dump(d, f)

        with open('test.pickle','rb') as f:
            d = pickle.load(f)

        self.assertEqual(d['2'], None)
        self.assertEqual(d['1'], {'1a':{'1ai':1,'1aii':2},'1b':{'1bi':3}})
        self.assertEqual(d['1','1a'], {'1ai':1,'1aii':2})
        self.assertEqual(d['1','1b'], {'1bi':3})
        self.assertEqual(d['1','1a','1ai'], 1)
        self.assertEqual(d['1','1a','1aii'], 2)
        self.assertEqual(d['1','1b','1bi'], 3)

    def test_convert_to_dict(self):
        d = starsmashertools.helpers.nesteddict.NestedDict({
            '1' : {
                '1a' : {
                    '1ai'  : 1,
                    '1aii' : 2,
                },
                '1b' : {
                    '1bi' : 3,
                },
            },
            '2' : None,
            'somekey' : None,
        })

        a = d.to_dict()
        self.assertTrue(isinstance(a, dict))
        self.assertFalse(isinstance(a, starsmashertools.helpers.nesteddict.NestedDict))
        self.assertEqual(len(a.keys()), 3)
        for key in ['1','2','somekey']: self.assertIn(key, a.keys())
        
        self.assertEqual(a['1'], {'1a' : {'1ai':1,'1aii' : 2}, '1b' : {'1bi':3}})
        self.assertEqual(a['1']['1a'], {'1ai':1,'1aii' : 2})
        self.assertEqual(a['1']['1a']['1ai'], 1)
        self.assertEqual(a['1']['1a']['1aii'], 2)
        self.assertEqual(a['1']['1b'], {'1bi':3})
        self.assertEqual(a['1']['1b']['1bi'], 3)
        self.assertEqual(a['2'], None)
        self.assertEqual(a['somekey'], None)

        a = starsmashertools.helpers.nesteddict.NestedDict(a)
        self.assertEqual(d['2'], None)
        self.assertEqual(d['1'], {'1a':{'1ai':1,'1aii':2},'1b':{'1bi':3}})
        self.assertEqual(d['1','1a'], {'1ai':1,'1aii':2})
        self.assertEqual(d['1','1b'], {'1bi':3})
        self.assertEqual(d['1','1a','1ai'], 1)
        self.assertEqual(d['1','1a','1aii'], 2)
        self.assertEqual(d['1','1b','1bi'], 3)
        
        

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            'test_keys',
            'test_values',
            'test_items',
            'test_branches',
            'test_stems',
            'test_leaves',
            'test_flowers',
            'test_getitem',
            'test_setitem',
            'test_list',
            'test_len',
            'test_contains',
            'test_iter',
            'test_clear',
            'test_copy',
            'test_deepcopy',
            'test_fromkeys',
            'test_get',
            'test_pop',
            'test_popitem',
            'test_reversed',
            'test_setdefault',
            'test_update',
            'test_comparisons',
            'test_del',
            'test_or_ior',
            'test_pickle',
            'test_convert_to_dict',
        ]

if __name__ == "__main__":
    unittest.main(failfast=True, testLoader=TestLoader())
