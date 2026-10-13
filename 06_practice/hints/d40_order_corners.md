Classic trick: TL has the smallest x+y, BR the largest; TR has the
smallest y-x, BL the largest. Fine up to ~30 degrees, but near 45
degrees two corners tie and it picks the same point twice. Sturdier:
sort by angle around the centroid (a consistent cycle), then rotate the
cycle to start at the smallest x+y. Try both.
