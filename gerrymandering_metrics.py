import numpy as np 


# political_leanings is dictionary with tuples of coordinates as keys, value is the political leaning, zero or one
# districting_plan is dictionary with tuples of coordinates as keys, value is the region.
def efficiency_gap(districting_plan, political_leanings):

	total_votes = len(political_leanings)

	# convert the districting plan into a dict 
	districts = {}
	for (key, value) in districting_plan.items():
		if value in districts.keys():
			districts[value] = districts[value] + [key]
		else:
			districts[value] = [key] 

	
	efficiency_gap = []

	# for each district calculate the wasted votes
	for district, counties in district.items():
		district_size = len(counties)
		votes = [political_leanings[county] for county in counties]

		party_1_votes=  (district_size - sum(votes))
		party_2_votes =  sum(votes)


		# if party 1 wins
		if party_1_votes >= party_2_votes:
			party_1_wasted_votes = party_1_votes - district_size/2
			party_2_wasted_votes = party_2_votes
		else:
			party_1_wasted_votes = party_1_votes
			party_2_wasted_votes = party_2_votes - district_size/2

		efficiency_gap = efficiency_gap + [party_1_wasted_votes - party_2_wasted_votes]
	

	return sum(abs(efficiency_gap))/total_votes






