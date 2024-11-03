import os
import json

import pymysql

def main():
    """
    아래와 같이 기본적으로 null string -> NULL 처리를 해줌.
    
    SET SQL_SAFE_UPDATES = 0;

    UPDATE cosmos.product 
    SET seller_spec = NULL
    WHERE seller_spec like 'null';

    SET SQL_SAFE_UPDATES = 1;
    """
    conn = pymysql.connect(
        # host=os.getenv('DB_HOST'), # 'localhost',
        host="localhost", # test: 'localhost',
        user='root',
        password='1234',
        db='cosmos',
        port=3306,
        charset='utf8mb4',
        collation='utf8mb4_general_ci',
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cursor:        
            cursor.execute("SELECT * FROM cosmos.product")
            result = cursor.fetchall()
            
            update_data = []
            for row in result:                
                try:
                    seller_spec = json.loads(row['seller_spec']) # None 처리.
                except: 
                    continue
                
                if not seller_spec: # DB에서 null이란 string을 위한 처리.
                    update_data.append((json.dumps(None, ensure_ascii=False, indent=4), row['prid']))
                    continue
                
                if not isinstance(seller_spec[0], list):                    
                    continue
                
                # 리팩토링 필요.
                restructured_packets = []
                
                for img_spec in seller_spec:  
                    if img_spec[0] == "":
                        continue
                                                      
                    default_packet = {
                        "img_str": "",
                        "bbox_text": [],
                        "our_topics": []
                    }
                
                    default_packet["img_str"] = img_spec[0]
                    if isinstance(img_spec[1], list):
                        default_packet['our_topics'] = img_spec[1]
                        default_packet["bbox_text"] = img_spec[2:]
                    else:
                        default_packet["bbox_text"] = img_spec[1:]
                    
                    restructured_packets.append(default_packet)
                
                update_data.append((json.dumps(restructured_packets, ensure_ascii=False, indent=4), row['prid']))                
            
            sql = """
                UPDATE cosmos.product SET seller_spec = %s WHERE prid = %s
            """
            
            cursor.executemany(sql, update_data)
            
            conn.commit()
            print(f"Updated {len(update_data)} records successfully.")
            
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()
                

if __name__ == "__main__":
    main()