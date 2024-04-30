"""

SNR Computing - Literature Search

Basal icequakes during changing subglacial water pressure beneath Gornergletscher, Switzerland
(von Fabian)

 - Fabian berechnet STA/LTA für Seismometer Daten
    - LTA duration: 800 ms (=8 sec)
    - STA duration: 80 ms (=0.8 sec)
    - the root-mean-square is computed in LTA and STA window.
    - when STA/LTA exceeds threshold of 10 (or 20, depending on data set), a trigger is set
    - an additional parameter n_trig is set. n_trig records have to be triggerend, in order to
      trigger an event.
 - this procedure makes also sense for triggering events on DAS data. (maybe with an additional parameter
   indicating how far two small triggers can be appart at most, to belong to one event)
 - STA/LTA ist letztendlich auch Berechnung von SNR value
 - Berechnung von SNR value nur sinnvoll, wenn sowohl auf denoisten als auch rohen Daten ein Signal
   vorhanden ist.

Further Computation of SNR values of DAS





"""