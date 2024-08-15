
In INDICATE wird für die Kurvenanpassung bei SAM eine
Implementierung von einer "Generic Levenberg-Marquardt routine (lmmin)"
von Joachim Wuttke (Forschungszentrum Juelich GmbH verwendet).

Zum Vergleichen der Kurvenanpassung mit dem "Standard-R-Paket" 'drc'
enthält die Datei r_lmcurve_ll5.c ein Binding für die Programmiersprache R.

Siehe Source-Dateien und COPYING.


Zum Kompilieren des Algorithmus bzw. des Bindings/Wrappers für R bitte das Shell-Skript ausführen:

> sh compile.sh

... oder direkt den Befehl:

> R CMD SHLIB r_lmcurve_ll5.c



ACHTUNG!
Die Kompilierung wurde nur unter Linux getestet.
Eventuell müssen vor der Kompilierung auf dem jeweiligen Betriebssystem
noch notwendige Software-Pakete installiert werden.
