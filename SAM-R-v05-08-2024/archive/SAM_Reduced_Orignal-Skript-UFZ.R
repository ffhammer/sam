require(drc)
pfad="C:\\Users\\foit\\UFZ\\3_Projekte\\31_Multipler Stress\\6_Plots"
###########################################################################################



#Hier eine neue PlateauRegel:
#etwas verrückt, aber vielleicht ganz nett:  

NullF=function(x,y){
x1=log10(x)
yEffect=(y[1]-y[2])/y[1]#Wie groß ist der erste Effekt (0 bis 1)
ShiftControl = 1 + ceiling(yEffect*10) #also Größenordnung 1 wie bisher auf jeden Fall, aber evtl mehr, in Abhängigkeit von erstem Effekt 
x1[1]=round(x1[2],0) - ShiftControl #Die Nullkonzentration auf Log-Skala festsetzen
X1=10^x1[1]
X1}

NullF2=function(x,y){
x1=log10(x);x1
yEffect=(y[1]-y[2])/y[1]#Wie groß ist der erste Effekt (0 bis 1)
ShiftControl = 2 + ceiling(yEffect*10);ShiftControl #also Größenordnung 1 wie bisher auf jeden Fall, aber evtl mehr, in Abhängigkeit von erstem Effekt 
xControlL=round(x1[2],0)- ShiftControl;xControlL
xControl=10^xControlL;xControl
xControl}

#--------------------------------------
#Funktionen zur Berechnung der Effekte:

B=NA;D=NA;E=NA;F=NA #parameter values on NA

Feffect = function(conc,B,C,D,E,F){C + (D-C)/(1+(conc/E)^B)^F}
Flc = function(effect,B,C,D,E,F){E * ((((D-C)/(effect-C))^(1/F) - 1)^(1/B))} 
#----------------------------------------------


############################################################################################

graphics.off() #alle offenen plot fenster schließen
# windows(3.2,3.2,rescale="fixed")
par(mfrow=c(1,1),mar=c(2.7, 2.7, 0.5, 1),lab=c(4,4,1),mgp=c(1.5, 0.3, 0),tck=0.01,cex=0.85)


#example data mit Stützpunkt bei 0.01:
conc=c(0,0.01,1,3,10,30,100);conc #Schadstoffkonzentration
effect=c(100,100,90,70,60,50,0);effect #Effekt Schadstoff ohne Umweltstress
conc2=conc
effect2=c(90,90,60,40,30,20,0);effect2 #Effekt Schadstoff und Umweltstress kombiniert

#example data ohne Stützpunkt bei 0.01:
conc=c(0,1,3,10,30,100);conc #Schadstoffkonzentration
effect=c(100,90,70,60,50,0);effect #Effekt Schadstoff ohne Umweltstress
conc2=conc
effect2=c(90,60,40,30,20,0);effect2 #Effekt Schadstoff und Umweltstress kombiniert

#Mit Plateauregel den Shift für alle Konzentrationen zum Plotten berechnen:
concShift = NullF2(conc,effect);concShift
concShift = 0.1
concPlot=conc+concShift;concPlot

#Zur Berechnung 1000 Konzentrationsstufen
concNPlot=10^seq(log10(min(concPlot)),log10(max(concPlot)),length=1000);concNPlot
concN=concNPlot-concShift;head(concN)

################################################################################
#Optional: Williams und Approx:

#Das wäre auch nicht schlecht als Option: WilliamsTransformation und Approximation der Zwischenpunkte (alles so wie im paper mit einem Knopf anstellbar)
#1. Williams-Transformation:
WilliamsF=function(vec){
  vec
  vecF=NULL
  steps=vec[1:(length(vec)-1)]-vec[2:length(vec)];steps
  count=rep(1,length(vec));count
  outlier=which(steps<0)[1];outlier

    while(length(na.omit(outlier))>0){
      #vec[outlier:(outlier+1)]=mean(vec[outlier:(outlier+1)]);vec #Das wäre ohne Wichtung, gerade STudie 19 wird optische besser wenn Wichtung angestellt....
      vec[outlier:(outlier+1)]=weighted.mean(vec[outlier:(outlier+1)],count[outlier:(outlier+1)]);vec
      vec=vec[-(outlier+1)];vec
      count[outlier]=count[outlier]+count[outlier+1];count
      count=count[-(outlier+1)];count
      steps=vec[1:(length(vec)-1)]-vec[2:length(vec)];steps
      outlier=which(steps<0)[1];outlier
    }#while Problem zu

  for(i in 1:length(vec)){
    vecF=c(vecF,rep(vec[i],count[i]))
  }
 vecF #return
}#function Ende


#Da eine Option ein- und ausschaltbar:
interpolation = 0
if(interpolation == 1){
#1. Williams anwenden:
effectT=WilliamsF(effect);effectT #Ohne Stress
effectT2=WilliamsF(effect2);effectT2 #Mit Stress
#2. Mit Approx echte Werte durch Zwischenwerte ersetzen:
#Geht hier leider nicht, da ja jetzt mit echten Werten parametrisiert wird... 
concA=10^approx(log10(conc), effectT, n = 10)$x ;concA #Für neue X-Werte die neuen Y-Werte
effectA=approx(log10(conc), effectT, n = 10)$y ;effectA #Für neue X-Werte die neuen Y-Werte
effect2A=approx(log10(conc), effectT2, n = 10)$y ;effect2A #Für neue X-Werte die neuen Y-Werte
conc=concA
effect=effectA
conc2=conc
effect2=effect2A
}

################################################################################
#Plots und SAM

YLIM=c(0,100)
COL=c("blue","red","green")

plot(log10(concPlot),effect,type="n",ylim=YLIM)
points(log10(concPlot),effect,col=COL[1],pch=19)
points(log10(concPlot),effect2,col=COL[2],pch=19)



# ================================================


#2. Dose-Response für Transformierte Daten ohne Stress:
mod = drm(effect ~ conc, fct=LL.5(c(NA,0,effect[1],NA,NA)));summary(mod)
P=mod$fct$fixed;P[which(is.na(P))]=mod$fit$par;P=round(P,2);P #ParameterWerte raussuchen
D=P[3];D #Wichtig für die SAM-Modellierung


#Line dazu plotten:
effectN=Feffect(concN,P[1],P[2],P[3],P[4],P[5]);effectN
lines(log10(concNPlot),effectN,col=COL[1])


#Effektkonzentrationen
lcEffects=D/100*c(90,50);lcEffects
lc=Flc(lcEffects,P[1],P[2],P[3],P[4],P[5]);lc
points(log10(lc+concShift),lcEffects,pch=6,col=COL[1])

# ================================================

#3. Dose-Response für Transformierte Daten mit Stress:
mod = drm(effect2 ~ conc2, fct=LL.5(c(NA,0,effect2[1],NA,NA)));summary(mod) 
P2=mod$fct$fixed;P2[which(is.na(P2))]=mod$fit$par;P2=round(P2,2);P2 #ParameterWerte raussuchen
D2=P2[3] #Wichtig für die Bestimmung des Umweltstresses s.u.
#Line dazu plotten:
effectN2=Feffect(concN,P2[1],P2[2],P2[3],P2[4],P2[5]);effectN2
lines(log10(concNPlot),effectN2,col=COL[2])

#Effektkonzentrationen
lcEffects2=D2/100*c(90,50);lcEffects2
lc2=Flc(lcEffects2,P2[1],P2[2],P2[3],P2[4],P2[5]);lc2
points(log10(lc2+concShift),lcEffects2,pch=6,col=COL[2])




#--------------------------
#Und jetzt SAM...
BETA=3.2

##1. Stress der Toxicant-Belastung:
head(concN)
head(effectN)
StressTox=qbeta(1-effectN/max(effectN), BETA, BETA, lower.tail = TRUE, log.p = FALSE);StressTox

##2. Stress nur für Umweltstress:
EffectEnv=1-(1/D*D2);EffectEnv #Möglichkeit 2: Differenz zwischen Maximalwerten nach Williams-Transformation
StressEnv=qbeta(EffectEnv, BETA, BETA, lower.tail = TRUE, log.p = FALSE)
StressEnv

##3. Stress Toxicant und Stress Umwelt addieren und gemeinsamen SAM-Effekt berechnen:
Stress=StressTox+StressEnv;Stress
Stress
SAM=(1-pbeta(Stress, BETA, BETA, lower.tail = TRUE, log.p = FALSE))*D;SAM
lines(log10(concNPlot),SAM,lty=2,lwd=2, col=COL[3])#HIERMIT PLOTTEN!!

#Berechnung von LC10 und LC50 dazu:
lcEffects3=max(SAM)/100*c(90,50);lcEffects3
WO1=which.min(abs(SAM-lcEffects3[1]));WO1 #hab die Auflösung der Konzentrationen auf 1000 hochgesetzt, s.o., könnte man ja auch noch höher..
WO2=which.min(abs(SAM-lcEffects3[2]));WO2 #
lc3=concNPlot[c(WO1,WO2)];lc3                 #was sind die Konzentrationen zum jeweiligen Effekt?
points(log10(lc3),lcEffects3,pch=6,col=COL[3])    #ich find das reicht...


#------------------
# savePlot(filename = file.path(pfad,"Schlenker_001_ohneStützpunkt"),type = "png")
# pdf("Kaarina-output.pdf")
#--------------------------
#EC-Values:

lc
lc2
lc3



