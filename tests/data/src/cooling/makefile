.SUFFIXES: .f

AR := ar ruv

# FC and FCFLAGS are defined in the StarSmasher makefile, using "export FF=" and
# "export FCFLAGS=" respectively

OBJS = nearby_particles.o get_ncooling2.o get_ncooling3.o \
	qtau.o init_cooling.o bilinear_interpolation.o \
	getOpacity.o tseval_f90.o init_icosahedron.o \
	kappa_and_T.o fluffy.o dense.o

COOLINGLIB = libcooling.a

all: $(COOLINGLIB)

$(COOLINGLIB): $(OBJS)
	/bin/rm -f $@
	$(AR) $@ $(OBJS)

%.o: %.f cooling.h
	$(FC) -c $(FCFLAGS) $<

clean:
	/bin/rm -rf *.o *.mod $(COOLINGLIB)

$(COOLINGLIB): cooling.h
