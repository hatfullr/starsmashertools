import starsmashertools
import os

directories = [
    os.path.join(starsmashertools.SOURCE_DIRECTORY, 'starsmashertools'),
    starsmashertools.DEFAULTS_DIRECTORY,
    starsmashertools.TEST_DIRECTORY
]

def get_files(src):
    with os.scandir(src) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks = False):
                yield from get_files(entry.path)
            elif entry.is_file(follow_symlinks = False):
                yield entry.path

def find_backtick_mismatches(_file):
    # Backticks will never appear across linebreaks
    import re
    reg1 = re.compile(r'\`{2}[^\`]+\`{2}[^\\s]', flags = re.M)
    reg2 = re.compile(r'(?<!\`)\`[^\`\n]+\`[^\\s]', flags = re.M)


    def check_line(line):
        if '`' not in line: return

        # Just ignore ```
        line = line.replace('```','')

        if not line.strip(): return
        
        doubles = line.split('``')
        if '``' in line:
            singles = line.replace('``','').split('`')
        else:
            singles = line.split('`')

        for double in doubles[1::2]:
            phrase = '``'+double+'``'
            idx = line.index(phrase) + len(phrase)
            if idx == len(line): continue
            if line[idx] in [' ','\\']: continue
            print('Offending character =', line[idx])
            print('Offending file =', _file)
            print('Offending line =', line)
            raise Exception("Backticks need to be followed by whitespace. Use an escaping \\ character to fix.")

        for single in singles[1::2]:
            phrase = '`' + single + '`'
            start = line.index(phrase)
            end = start + len(phrase)

            if end == len(line): continue
            if line[start - 1: end] == '``'+single+'``': continue
            if line[end] in [' ', '\\']: continue
            print('Offending phrase =', line[start - 1: end + 1])
            print('Offending character =', line[end])
            print('Offending file =', _file)
            print('Offending line =', line)
            raise Exception("Backticks need to be followed by whitespace. Use an escaping \\ character to fix.")

    
    try:
        with open(_file, 'r') as f:
            for line in f:
                check_line(line.strip())
    except UnicodeDecodeError: return 0

if __name__ == '__main__':
    for directory in directories:
        for _file in get_files(directory):
            basename = os.path.basename(_file)
            # Ignore emacs temp files
            if basename.startswith('#') or basename.endswith('#'): continue
            if basename.endswith('~'): continue
            find_backtick_mismatches(_file)
                
        

    
