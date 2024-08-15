# ===========================================================================================
# Überarbeitete Implementierung von SAM für R von M.Foit, 5.8.2024
#
# Das Original-Skript vom UFZ liegt im Ordner './archive'
# Das Original-Skript war nie so richtig fertig und poliert.
# Für die Umsetzung von SAM in INDICATE wurde daher zuerst dieses überarbeitete Skript entwickelt.
# Das Skript bildet also die Grundlage für die  Umsetzung von SAM in INDICATE.
#
# Bei der Hinzufügung der Interpolationspunkte hat es, in Absprache mit dem UFZ, einige Abweichungen vom Original-Skript gegeben,
# um die Kurvenanpassung insgesamt 'etwas robuster' für verschiedene Datensätze zu machen.
#
# Da für INDICATE die Kurvenpassung aus dem drc-Paket in R nicht zur Verfügung stand
# wurde hier eine Vergleichsmöglichkeit mit einem Open-Source-Kurvenpassungsalgorithmus eingebaut (lmcurve/lmfit von Joachim Wuttke)
# Siehe hierzu Konstante 'CURVE_FITTING', Funktion 'll5_parameters' und das Verzeichnis './r-lmcurve-ll5'
# ===========================================================================================



# CONSTANTS
# ***********************
BETA=3.2					# beta für SAM
CONC0_MAX_DY = 5.0/100;		# wie weit darf die Kurve vom Null-Effekt abweichen [Prozent/100]
CONC0_MIN_EXP = -100;		# wie klein darf die Null-Konzentration sein [Exponent zu Basis 10]
LINEAR_INTER_STEPS = 10		# fixed number of interpolation steps
CURVE_FITTING = 'drc'		# Gewählter Kurvenanpassungsalgorithmus ('drc', oder 'lmcurve' für Anpassung wie in INDICATE. Siehe weiter unten.)

FILENAME_PDF = paste0("output_", CURVE_FITTING, ".pdf")


# LAYOUT
# ***********************
pdf(FILENAME_PDF, paper="a4r")
par(mfrow=c(2,2),mar=c(2.7, 2.7, 5.7, 2.7),lab=c(4,4,1),mgp=c(1.5, 0.3, 0),tck=0.01,cex=0.6)



# LL5-FUNCTION
# ***********************
ll5 = function(conc, b, c, d, e, f) {
	c + (d-c) / (1 + (conc/e)^b)^f
}

ll5_inv = function(surv, b, c, d, e, f) {
	e * ((((d-c)/(surv-c) )^(1/f) - 1)^(1/b))
}


# LL5-APPROXIMATION / Kurvenanpassung
# ********************************
ll5_parameters = NULL;

if(CURVE_FITTING == 'drc') {
	# [A] Anpassung mit drc
	suppressPackageStartupMessages({
		require(drc)
	})

	ll5_parameters = function (conc, surv, b, c, d, e, f) {
		p = NA

		tryCatch( {
			mod = drm(surv ~ conc, fct=LL.5(c(b,c,d,e,f)))
			p = mod$fct$fixed
			p[which(is.na(p))] = mod$fit$par
		}, error = function(e) {
			p = c(1,1,1,1,1)
		} )

		list(
			b=p[1],c=p[2],d=p[3],e=p[4],f=p[5]
		)
	}
} else if(CURVE_FITTING == 'lmcurve') {
	# [B] Anpassung wie in INDICATE ('lmcurve')!
	# Muss vor der Nutzung erst kompiliert werden.
	# Siehe hierzu das Verzeichnis './r-lmcurve-ll5'

	source("r-lmcurve-ll5/lmcurve_ll5.R", chdir = T)

	ll5_parameters = function (conc, surv, b, c, d, e, f) {
		p = lmcurve_ll5(conc, surv, b, c, d, e, f)

		if(is.na(p$b)) p$b = 1
		if(is.na(p$c)) p$c = 1
		if(is.na(p$d)) p$d = 1
		if(is.na(p$e)) p$e = 1
		if(is.na(p$f)) p$f = 1
		p
	}

}




# BETA-FUNCTIONS
# ***********************
qbet = function(x) {
	qbeta(x, BETA, BETA, lower.tail = TRUE, log.p = FALSE)
}

pbet = function(x) {
	pbeta(x, BETA, BETA, lower.tail = TRUE, log.p = FALSE)
}


# LIN INTERP./INVERSE
# ***********************
linear_inv = function(vec_x, vec_y, y) {
	stopifnot( (length(vec_x) == length(vec_y)) && (length(vec_x)>1) )

	for(i in 1:(length(vec_x)-1)) {
		j = i+1;
		if(
				((vec_y[i] <= y) && (vec_y[j] >= y))
			||	((vec_y[j] <= y) && (vec_y[i] >= y))
		) {
			return( (y - vec_y[i]) * (vec_x[j]-vec_x[i]) / (vec_y[j]-vec_y[i]) + vec_x[i] )
		}
	}
	return (NA)
}


# NULL-CONCENTRATION
# ***************************
find_c0 = function(conc, pa, pb) {
	c0 = NA

	tryCatch( {
		ya0 = pa$d
		yb0 = pb$d

		conc0_max_exp = floor(log10(conc[2]))

		for(i in conc0_max_exp : CONC0_MIN_EXP) {
			c0 = 10^i

			ya = ll5(c0, pa$b, pa$c, pa$d, pa$e, pa$f)
			yb = ll5(c0, pb$b, pb$c, pb$d, pb$e, pb$f)
			ysam = pa$d * (1 - pbet(qbet(1 - ya/pa$d) + qbet(1 - (pb$d / pa$d))))

			dya_relative = (ya0-ya) / ya0;
			dyb_relative = (yb0-yb) / yb0;
			dysam_relative = (yb0-ysam) / yb0;

			# % bezogen auf "Maximalwert" (= Parameter d)
			if(
				(abs(dya_relative) < CONC0_MAX_DY) &&
				(abs(dyb_relative) < CONC0_MAX_DY) &&
				(abs(dysam_relative) < CONC0_MAX_DY)
			)
				return(c0)
		}

	}, error = function(e) {
		print("error finding c0")
		print(e)
		c0 = 10^(log10(conc[1]) - 1)
	});

	return (c0)
}




# TRANSFORMATIONS
# ***************************
transform_none = function(conc, surv) {
	list(conc=conc, surv=surv)
}


transform_linear_interpolation = function(conc, surv) {
	c0 = conc[2] / 2 	# linker Rand des Intervals über das dann interpoliert wird
	e0 = surv[2]

	conc_c = c(c0, conc[-1])
	surv_c = c(e0, surv[-1])

	points = approx( log10(conc_c), surv_c, n = LINEAR_INTER_STEPS )

	list(
		conc=c(conc[1], 10^points$x),
		surv=c(surv[1], points$y)
	)
}


transform_williams = function(conc, surv) {
	vec=surv # !
	vecF=NULL
	steps=vec[1:(length(vec)-1)]-vec[2:length(vec)]
	count=rep(1,length(vec))
	outlier=which(steps<0)[1]

	while(length(na.omit(outlier))>0) {
		vec[outlier:(outlier+1)]=weighted.mean(vec[outlier:(outlier+1)],count[outlier:(outlier+1)])
		vec=vec[-(outlier+1)]
		count[outlier]=count[outlier]+count[outlier+1]
		count=count[-(outlier+1)]
		steps=vec[1:(length(vec)-1)]-vec[2:length(vec)]
		outlier=which(steps<0)[1]
	}

	for(i in 1:length(vec)){
		vecF=c(vecF,rep(vec[i],count[i]))
	}

	list(
		conc=conc,
		surv=vecF # !
	)
}


transform_williams_and_linear_interpolation = function(conc, surv) {
	data = list(conc=conc, surv=surv)
	data = transform_williams(data$conc, data$surv);
	data = transform_linear_interpolation(data$conc, data$surv)
	data
}



# PLOT
# ***************************
create_plot = function(name, conc, surv_a, surv_b, transform_name, f=NA) {
	transform = get(paste("transform_", transform_name, sep=""))

	get_properties = function(surv) {
		transformed = transform(conc, surv)
		conc_t = transformed$conc
		surv_t = transformed$surv

		p = ll5_parameters(conc_t, surv_t, NA, 0, surv_t[1], NA, f)

		lc_surv = p$d/100 * c(90,50)

		list(
			b=p$b, c=p$c, d=p$d, e=p$e, f=p$f,
			lc=list(
				conc = ll5_inv(lc_surv, p$b, p$c, p$d, p$e, p$f),
				surv = lc_surv
			),
			conc_t=conc_t,
			surv_t=surv_t
		)
	}

	pa = get_properties(surv_a)
	pb = get_properties(surv_b)

	conc_0 = find_c0(conc, pa, pb)
	conc_n = max(conc)
	conc = sapply(conc, function(c) { if(c) c else conc_0 }) # replaces "all 0" by conc0 and OVERWRITES conc

	# data lines
	conc_line = 10^seq(log10(conc_0),log10(conc_n),length=100)

	# - approximated
	surv_line_a = ll5(conc_line, pa$b, pa$c, pa$d, pa$e, pa$f)
	surv_line_b = ll5(conc_line, pb$b, pb$c, pb$d, pb$e, pb$f)

	# - sam
	stress_a = qbet(1 - surv_line_a/pa$d)
	stress_b = qbet(1 - (pb$d / pa$d))
	surv_line_sam = (1 - pbet(stress_a + stress_b)) * pa$d

	lc_sam_surv = pb$d/100 * c(90,50)
	lc_sam_conc = sapply(lc_sam_surv, function(surv) {linear_inv(conc_line, surv_line_sam, surv)})


	# - EA (Effect Addition)
	surv_line_ea = (pb$d/pa$d) * surv_line_a;

	# - CA (Concentration Addition)
	conc_env_ca = pa$e * ( (pa$d/pb$d)^(1/pa$f)-1 )^( 1/pa$b )
	surv_line_ca = ll5(conc_line+conc_env_ca, pa$b, 0, pa$d, pa$e, pa$f)



	# --------------------------------------------------------

	# plot
	plot(c(conc_0, conc_n), c(NA,NA), log="x", xaxt="n", type="n", ylim=c(0, max(surv_a, surv_b)), xlab="Concentration", ylab="Survival")
	title(main = paste("Transform: ", transform_name, " ( f:",f, ")"), line=1)

	mtext(paste0(name, " (", CURVE_FITTING, ")"), outer=TRUE,  cex=1, line=-2)

	conc_exp_0 = floor(log10(conc_0))
	conc_exp_n = ceiling(log10(conc_n))
	conc_exp_seq = seq(conc_exp_0, conc_exp_n, by=ceiling((conc_exp_n - conc_exp_0)/10) )
	axis(1, at=10^conc_exp_seq, labels=sapply(conc_exp_seq, function(exp) { parse(text=paste('10^',exp)) }))

	# - points
	points(conc, surv_a, col="blue", pch=19)
	points(conc, surv_b, col="red", pch=19)

	# - points of transformed data
	points(pa$conc_t, pa$surv_t, col=rgb(0,0,0,0.35), pch=4)
	points(pb$conc_t, pb$surv_t, col=rgb(0,0,0,0.35), pch=4)

	# - lines, ll5
	lines(conc_line, surv_line_a, col="blue", lty=1)
	lines(conc_line, surv_line_b, col="red", lty=1)

	# - lines, predicted
	lines(conc_line, surv_line_sam, col="red", lty=2)
	lines(conc_line, surv_line_ea, col="purple", lty=2)
	lines(conc_line, surv_line_ca, col="black", lty=2)


	# - lc-values
	points(pa$lc$conc, pa$lc$surv, pch=6, col="blue")
	points(pb$lc$conc, pb$lc$surv, pch=6, col="red")
	points(lc_sam_conc, lc_sam_surv, pch=6, col="red")
}


create_plots = function(name, conc, surv_a, surv_b) {
	create_plot(name, conc, surv_a, surv_b, "none");
	create_plot(name, conc, surv_a, surv_b, "williams_and_linear_interpolation");
	create_plot(name, conc, surv_a, surv_b, "none", f=1);
	create_plot(name, conc, surv_a, surv_b, "williams_and_linear_interpolation", f=1);
}


get_prediction = function(name, conc, surv_a, surv_b, transform_name, f=NA) {
	transform = get(paste("transform_", transform_name, sep=""))

	get_properties = function(surv) {
		transformed = transform(conc, surv)
		conc_t = transformed$conc
		surv_t = transformed$surv

		p = ll5_parameters(conc_t, surv_t, NA, 0, surv_t[1], NA, f)

		lc_surv = p$d/100 * c(90,50)

		list(
			b=p$b, c=p$c, d=p$d, e=p$e, f=p$f,
			lc=list(
				conc = ll5_inv(lc_surv, p$b, p$c, p$d, p$e, p$f),
				surv = lc_surv
			),
			conc_t=conc_t,
			surv_t=surv_t
		)
	}

	pa = get_properties(surv_a)
	pb = get_properties(surv_b)

	conc_0 = find_c0(conc, pa, pb)
	conc_n = max(conc)
	conc = sapply(conc, function(c) { if(c) c else conc_0 }) # replaces "all 0" by conc0 and OVERWRITES conc

	# data lines
	conc_line = 10^seq(log10(conc_0),log10(conc_n),length=100)

	# - approximated
	surv_line_a = ll5(conc_line, pa$b, pa$c, pa$d, pa$e, pa$f)
	surv_line_b = ll5(conc_line, pb$b, pb$c, pb$d, pb$e, pb$f)

	# - sam
	stress_a = qbet(1 - surv_line_a/pa$d)
	stress_b = qbet(1 - (pb$d / pa$d))
	surv_line_sam = (1 - pbet(stress_a + stress_b)) * pa$d

	lc_sam_surv = pb$d/100 * c(90,50)
	lc_sam_conc = sapply(lc_sam_surv, function(surv) {linear_inv(conc_line, surv_line_sam, surv)})


	# - EA (Effect Addition)
	surv_line_ea = (pb$d/pa$d) * surv_line_a;

	# - CA (Concentration Addition)
	conc_env_ca = pa$e * ( (pa$d/pb$d)^(1/pa$f)-1 )^( 1/pa$b )
	surv_line_ca = ll5(conc_line+conc_env_ca, pa$b, 0, pa$d, pa$e, pa$f)


	list(
			Concentration = conc_line,
			Survival_A = surv_line_a,
			Survival_B = surv_line_b,
			SAM = surv_line_sam,
			EA = surv_line_ea,
			CA = surv_line_ca,
			LC_SAM_Concentration = lc_sam_conc,
			LC_SAM_Survival = lc_sam_surv
		)
	}





# ====================================================================
# TEST SAMPLE DATA
# ====================================================================

# Test mit verschiedenen Datensätze.
# Vor allen Dingen um die Kurvenanpassung vergleichen zu können

# conc = c(0,1,3,10,30,100)
# create_plots("Sample Data", conc, c(100,90,70,60,50,0), c(90,60,40,30,20,0));

# # Charlotte Ch24
# conc = c(0,5,10,20,40,80)
# create_plots("Charlotte Ch24", conc, c(100,100,100,100,55,35), c(98.3,98.3,98.3,45,40,10));

# # Charlotte Cd24
# conc = c(0,0.15,0.56,2.3,7.1)
# create_plots("Charlotte Cd24", conc, c(100,100,80,20,0), c(98.3,90,60,10,0));

# # Esfen with and wo Stressor
# conc = c(0, 0.1, 0.316, 1)
# create_plots("Esfen with and wo Stressor", conc, c( 100, 93.333, 73.3, 73.3 ), c(93.3, 60, 13, 6.6));

# # Prochlo + Esfen with and wo Stressor
# conc = c(0, 0.1, 0.316, 1)
# create_plots("Prochlo + Esfen with and wo Stressor", conc, c( 100, 93.3, 73.3, 86.6 ), c(60, 35.7, 35.7, 0));

# # Stressor Groups (attached with same email as "Prochlo + Esfen" )
# conc = c(0, 0.1, 0.316, 1)
# create_plots("Stressor Groups (same email as 'Esfen')", conc, c( 93, 60, 13, 6.6 ), c(60, 35.7, 35.7, 0));


# conc = c(0,0.0001,0.001,0.01,0.1,0.32,1,3.2)
# create_plots("Probleme-Email-02-11-2017", conc, c(1, 1, 0.96774194, 0.95555555, 0.84444444, 0.62222222, 0.34090909, 0), c(0.52,0.35,0.186,0.0875,0.034,0,0,0));



