import matplotlib.figure

class ExampleFigure(matplotlib.figure.Figure):
    name = 'ExampleFigure'
    _is_example = True # Excludes this class from the actual code



if __name__ == '__main__':
    import starsmashertools.mpl
    import matplotlib.pyplot as plt

    starsmashertools.mpl._is_example = True
    
    fig, ax = starsmashertools.mpl.subplots(FigureClass = 'ExampleFigure')

    plt.show()
