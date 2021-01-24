#!/bin/bash

#for the complete dataset ;)
#rm -f ./data/WDI*
#rm ./data/ip_jrn_art.csv
#rm ./data/sp_pop_totl
#load WDI data
#wget --directory-prefix=./data http://databank.worldbank.org/data/download/WDI_csv.zip
#unzip ./data/WDI_csv.zip -d ./data

declare -a indicators=("IC.ISV.DURS" "SE.TER.ENRL.TC.ZS" "IP.PAT.RESD" "IP.PAT.NRES" "SE.PRM.ENRL.TC.ZS" "SE.SEC.ENRL.TC.ZS" "SE.PRE.ENRL.TC.ZS" "IC.REG.DURS" "IC.REG.PROC" "GB.XPD.RSDV.GD.ZS" "FR.INR.RINR" "IC.TAX.PRFT.CP.ZS" "SP.POP.1564.TO.ZS" "IC.BUS.NDNS.ZS" "CM.MKT.LCAP.GD.ZS" "IT.NET.USER.ZS" "HD.HCI.OVRL" "SE.XPD.TOTL.GD.ZS" "NY.GNP.PCAP.CN" "NY.GNP.PCAP.CD" "BM.KLT.DINV.WD.GD.ZS" "BX.KLT.DINV.WD.GD.ZS" "NE.CON.TOTL.ZS" "SP.DYN.TFRT.IN" "IC.BUS.DFRN.XQ" "IC.REG.COST.PC.ZS" "EN.ATM.CO2E.PC")

for i in "${indicators[@]}"
do
   echo $i
   wget --output-document ./data/tmp.zip https://api.worldbank.org/v2/en/indicator/$i?downloadformat=csv
   unzip ./data/tmp.zip -d ./data
   rm ./data/tmp.zip

    mv ./data/API_*.csv ./data/data
    mv ./data/Metadata*.csv ./data/metadata
done




