import starsmashertools.helpers.string
import unittest
import os
import basetest

class TestString(basetest.BaseTest):
    def test_is_index_in_literal(self):


        tests = {
            '012' : [False]*3,
            '"012' : [False]*3,
            '0"12' : [False]*3,
            '01"2' : [False]*3,
            '012"' : [False]*3,
            
            "'012" : [False]*3,
            "0'12" : [False]*3,
            "01'2" : [False]*3,
            "012'" : [False]*3,
            
            '\"012' : [False]*3,
            '0\"12' : [False]*3,
            '01\"2' : [False]*3,
            '012\"' : [False]*3,

            "\'012" : [False]*3,
            "0\'12" : [False]*3,
            "01\'2" : [False]*3,
            "012\'" : [False]*3,
            
            '\\"012' : [False]*4,
            '0\\"12' : [False]*4,
            '01\\"2' : [False]*4,
            '012\\"' : [False]*4,

            "\\'012" : [False]*4,
            "0\\'12" : [False]*4,
            "01\\'2" : [False]*4,
            "012\\'" : [False]*4,

            '\\\"012' : [False]*5,
            '0\\\"12' : [False]*5,
            '01\\\"2' : [False]*5,
            '012\\\"' : [False]*5,

            "\\\'012" : [False]*5,
            "0\\\'12" : [False]*5,
            "01\\\'2" : [False]*5,
            "012\\\'" : [False]*5,

            '""012' : [False,False,False,False,False],
            '"0"12' : [False,True,False,False,False],
            '"01"2' : [False,True,True,False,False],
            '"012"' : [False,True,True,True,False],
            '0"12"' : [False,False,True,True,False],
            '01"2"' : [False,False,False,True,False],
            '012""' : [False,False,False,False,False],

            "''012" : [False,False,False,False,False],
            "'0'12" : [False,True,False,False,False],
            "'01'2" : [False,True,True,False,False],
            "'012'" : [False,True,True,True,False],
            "0'12'" : [False,False,True,True,False],
            "01'2'" : [False,False,False,True,False],
            "012''" : [False,False,False,False,False],

            '"0\'\"12' : [False,True,True,False,False],
            '"\'"012': [False,True,False,False,False,False],
        }

        for string, results in tests.items():
            for i, result in enumerate(results):
                self.assertEqual(
                    starsmashertools.helpers.string.is_index_in_literal(
                        string, i,
                    ),
                    result,
                    msg = repr(string) + ', ' + str(i),
                )

    def test_strip_inline_comment(self):
        self.assertEqual(
            starsmashertools.helpers.string.strip_inline_comment(
                'Hello # ignore this',
                '#',
            ),
            'Hello ',
        )

        self.assertEqual(
            starsmashertools.helpers.string.strip_inline_comment(
                'Hello "# ignore this"',
                '#',
            ),
            'Hello "# ignore this"',
        )

        self.assertEqual(
            starsmashertools.helpers.string.strip_inline_comment(
                'Hello "# ignore this" # actually',
                '#',
            ),
            'Hello "# ignore this" ',
        )

        self.assertEqual(
            starsmashertools.helpers.string.strip_inline_comment(
                "Hello '# ignore this'",
                '#',
            ),
            "Hello '# ignore this'",
        )

        self.assertEqual(
            starsmashertools.helpers.string.strip_inline_comment(
                "Hello '# ignore this' # actually",
                '#',
            ),
            "Hello '# ignore this' ",
        )
                

if __name__ == '__main__':
    unittest.main(failfast = True)
