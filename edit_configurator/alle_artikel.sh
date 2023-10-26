#!/bin/bash
set -x
for article in 111520 112150 116490 116585 118104; do
  sed "s/118103/$article/g" 118103.json > ${article}.json
done

