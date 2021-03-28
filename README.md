# kopen-of-huren
Huis kopen of huren? Wat levert meer op?


Al zolang ik me kan herinneren roept iedereen om mij heen dat een huis kopen absoluut een goede investering is.
In eerste instantie lijkt dit zeer aannemelijk, vooral door de zeer lage hypotheek die men nu kan afsluiten (≈2.0% voor 30 jaar vast).

Wat ik wil vergelijken is, wat beter is voor mijn portomonee na N jaar:
- een huis kopen met een hypotheek van €`X` per maand
- een huis huren voor €`Y` per maand (waar `X>Y`) en €`(X-Y`) (de rest) maandelijks investeren in de aandelenmarkt

In een simpele berekening kwam ik erachter dat het niet helemaal duidelijk is wat het beste is.
Het is namelijk sterk afhankelijk van hoeveel je huis in waarde stijgt en hoeveel de aandelenmarkt stijgt.
Bij een eerste benadering lijkt het dat als `aandelenmarkt_stijging_pct` ≈ `huis_waarde_stijging_pct`, een huiskopen absoluut een groot voordeel oplevert.
Echter, als de aandelenmarkt het bijvoorbeeld 2% beter doet (wat **zeer** aannemelijk is), is het niet helemaal duidelijk.

"The Devil is in de details," dus het is mijn plan om een realistische benadering te doen, waarbij ik gebruik maak van:

- historische groei in de huizen- en aandelenmarkt sinds 1996 t/m 2021
- hyptotheekrenteaftrek
- vermogensbelasting
- huis onderhouds kosten
- een annuïteitenhypotheek
- WOZ belasting
- loonheffing
- jaarinkomen
- overdrachtsbelasting
- statistiek, Monte Carlo simulaties, en meer!

Er zijn een aantal parameters die hierbij relevant zijn:

| parameter | waarde | variabel |
| --- | --- | --- |
| huis prijs | €X | ja |
| hypotheekrente | 2.0% | nee |
| hypotheek duur | 30 jaar | nee |
| hypotheek soort | annuïteitenhypotheek | nee |
| vaste kosten van een huiseigenaar | 1% huiswaarde per jaar | nee |
| huur huis | €Y per maand | ja |
| gemiddelde waardestijging huis | 5% per jaar | ja |
| gemiddelde waardestijging aandelenmarkt | 7% per jaar | ja |
| jaarinkomen | €50k | ja |
| (studie)schulden | €20k | ja |