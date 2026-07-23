package postgres

import "time"

var hcmLocation = time.FixedZone("Asia/Ho_Chi_Minh", 7*60*60)

func timeNow() time.Time              { return time.Now().In(hcmLocation) }
func seconds(value int) time.Duration { return time.Duration(value) * time.Second }
