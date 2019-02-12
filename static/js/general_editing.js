function explodeMulti(jsnMulti) {
    if (jsnMulti.type.substring(0,5)!="Multi") {
        alert("Geometry is not multipart");
    } else {
        var features=[];
        var type = jsnMulti.type.substring(5)
        for (var i=0;i<jsnMulti.coordinates.length;i++) {
            var feature = {type:type, coordinates:jsnMulti.coordinates[i]};
            features.push(feature);
        }
        return features;
    }
}

function mergeLyrEdit(lyrEdit) {
    var jsnEdited=lyrEdit.toGeoJSON();
    var arCoordinates=[];
    var type="Multi"+jsnEdited.features[0].geometry.type;
    for (var i=0;i<jsnEdited.features.length;i++) {
        var coordinates = jsnEdited.features[i].geometry.coordinates;
        arCoordinates.push(coordinates);
    }
    return {type:type, coordinates:arCoordinates};
}

function insertRecord(jsn, callback){
    delete jsn.id;
    $.ajax({
        url:'php/insert_record.php',
        data:jsn,
        type:'POST',
        success: function(response) {
            if (response.substr(0,5)=="ERROR") {
                alert(response);
            } else {
                alert("New record added to "+jsn.tbl);
                callback();
            }
        }, 
        error: function(xhr, status, error) {
            alert("AJAX Error: "+error);
        }
    })
}

function updateRecord(jsn, callback) {
    $.ajax({
        url:'php/update_record.php',
        data:jsn,
        type:'POST',
        success: function(response) {
            if (response.substr(0,5)=="ERROR") {
                alert(response);
            } else {
                alert("Record "+jsn.id+" in "+jsn.tbl+" updated");
                callback();
            }
        }, 
        error: function(xhr, status, error) {
            alert("AJAX Error: "+error);
        }
    })
}

function deleteRecord(tbl, id, callback) {
    $.ajax({
        url:'php/delete_record.php',
        data:{tbl:tbl, id:id},
        type:'POST',
        success: function(response) {
            if (response.substr(0,5)=="ERROR") {
                alert(response);
            } else {
                alert("Record "+id+" deleted from "+tbl);
                callback();
            }
        }, 
        error: function(xhr, status, error) {
            alert("AJAX Error: "+error);
        }
    });
}

