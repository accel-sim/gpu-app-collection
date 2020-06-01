#!/bin/bash
for i in `ls .`
do
	if [ -d $i ]; then
	if [ -f $i/powerdatafile ]; then
		echo -n $i",";
		awk '{print $1|"xargs"}' $i/powerdatafile |
		awk -F ',' ' BEGIN {
				
				thresh=5;
				neg_thresh=-5;
				s=0; 
				count=0;
				start=0;
				temp_avg=0;
				max_avg=0;
				printed=0;
			}
			{	for (d=1; d<=NF; d++) {
					if(count==0){
						start=d;
						s=$d;
						count=1;
						temp_avg=s/count;		
					}else{									
						temp_avg=s/count;
						diff=$d-temp_avg;
						if( diff >= 0){						
							if(diff > thresh){  		
								if(count > 100){					
									if(temp_avg > max_avg)
									{                                	                                        
										max_avg=temp_avg;
										printed=1;
									}
								}
								start=0;
								count=0;
								s=0;								
								temp_avg=0;
							}else{
								s=s+$d;
								count++;
							}                                                               
						}else{
							if(diff < neg_thresh){
								if(count > 100){                                           
                                                                        if(temp_avg > max_avg){
                                                                               max_avg=temp_avg;
										printed=1;
                                                                        }
								
								}
								start=0;
								count=0;
								s=0;	
								temp_avg=0;					
							}else{
								s=s+$d;
								count++;
							}     						
						}
						
					}
				}
			} 
			
			END { 
				if(printed==1){
					print max_avg
				}else{
					print temp_avg
				}
			}'
	fi
	fi
done


