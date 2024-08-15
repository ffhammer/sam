
# Binding of lmcurve/lmfit by Joachim Wuttke, Forschungszentrum Juelich GmbH for the R programming language.
# See also COPYING and README.txt

# Author: Marco Foit


dyn.load("./r_lmcurve_ll5.so")

lmcurve_ll5 = function(x, y, b=NA, c=NA, d=NA, e=NA ,f=NA) {
	stopifnot(length(x) == length(y))

	n = length(x)

	r = .C(
		"r_lmcurve_ll5",
		x = as.double(x),
		y = as.double(y),
		n = as.integer(n),

		b=as.double(b),
		c=as.double(c),
		d=as.double(d),
		e=as.double(e),
		f=as.double(f),
		NAOK=TRUE
	)

	list(
		b=r$b,
		c=r$c,
		d=r$d,
		e=r$e,
		f=r$f
	)
}
