
# If we have Matplotlib, prefer using the Tk backend (seems it's faster)
try:
    import matplotlib
    matplotlib.use('tkagg', force = False)
except: pass
