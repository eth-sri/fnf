#!/bin/bash

mkdir -p logs/crime
mkdir -p logs/law
mkdir -p logs/health
mkdir -p logs/adult
mkdir -p logs/compas

mkdir -p logs/crime/autoreg
mkdir -p logs/crime/gmm
mkdir -p logs/crime/flow

mkdir adult
mkdir compas
mkdir lawschool
mkdir ../data
mkdir ../data/health
cd ../data/health
wget https://foreverdata.org/1015/content/HHP_release3.zip --no-check-certificate
unzip HHP_release3.zip
