function openSubScreen(scrn){
    $(".modal").hide();
    $("#"+scrn).show();
}

function randomizePos(e){
    var offsetX = Math.random()*0.00500-0.0025;
    var offsetY = Math.random()*0.00500-0.0025;
    e.latitude=e.latitude+offsetY;
    e.longitude=e.longitude+offsetX;
    e.latlng=L.latLng([e.latitude, e.longitude]);
    return e;
}

//function randomizePos(e){
//    return e;
//}

function startAutolocate(){
    $("#btnAutolocate").html("On");
    storeSettings();
    clearInterval(intAutolocate);
    intAutolocate=setInterval(function(){
        if (mrkCurrentLocation) {
            mrkCurrentLocation.remove();
        }
        if ($("#btnFilter").html()=="On") {
            var flt = $("#numFilter").val();
        } else {
            var flt = 100000;
        }
        if (posCurrent.accuracy<flt) {
            var radius = Math.min(200, posCurrent.accuracy/2)
            radius = Math.max(10, radius)
            mrkCurrentLocation = L.circle(posCurrent.latlng, {radius:radius}).addTo(mymap);
            mymap.setView(posCurrent.latlng, 17);
        }
    }, $("#numAutolocate").val()*1000)
}

function stopAutolocate(){
    $("#btnAutolocate").html("Off");
    storeSettings();
    clearInterval(intAutolocate);
}

function startBreadcrumbs(){
    $("#btnBreadcrumbs").html("On");
    storeSettings();
    clearInterval(intBreadcrumbs);
    addBreadcrumb();
    intBreadcrumbs=setInterval(function(){
        if ($("#btnFilter").html()=="On") {
            var flt = $("#numFilter").val();
        } else {
            var flt = 100000;
        }
        if (posCurrent.accuracy<flt) {
            addBreadcrumb();
        }
    }, $("#numBreadcrumbs").val()*1000)
    clearInterval(intInfo);
    intInfo = setInterval(function(){
        populateInfo();
    }, $("#numBreadcrumbs").val()*1000)
}

function stopBreadcrumbs(){
    $("#btnBreadcrumbs").html("Off");
    storeSettings();
    clearInterval(intBreadcrumbs);
}

function addBreadcrumb(){
    if (posCurrent) {
        var radius = Math.min(200, posCurrent.accuracy/2)
        radius = Math.max(10, radius)
        var mrkBreadcrumb = L.circle(posCurrent.latlng, {radius:radius, color:'green'});
        mrkBreadcrumb.bindPopup("<h4>"+L.stamp(mrkBreadcrumb)+"</h4>Time: "+returnTimeFromUTC(posCurrent.timestamp)+"<br>Accuracy: "+posCurrent.accuracy+" m");
        lyrBreadcrumbs.addLayer(mrkBreadcrumb);
        populatePoints();
    }
}

function populateInfo(){
    if (posCurrent) {
        $(".info_cur_acc").html(posCurrent.accuracy.toFixed(1));
        if (isNaN(posCurrent.altitude)) {
            posCurrent.altitude="NA";
        } else {
            posCurrent.altitude=posCurrent.altitude.toFixed(1);
        }
        $("#info_cur_lat").val(posCurrent.latitude.toFixed(5));
        $("#info_cur_lng").val(posCurrent.longitude.toFixed(5));
        $("#info_cur_alt").val(posCurrent.altitude);
        $("#info_cur_tm").val(returnTimeFromUTC(posCurrent.timestamp));

        if (posPrevious) {
            $("#info_prv_lat").val(posPrevious.latitude.toFixed(5));
            $("#info_prv_lng").val(posPrevious.longitude.toFixed(5));
            $("#info_prv_alt").val(posPrevious.altitude);
            $("#info_prv_tm").val(returnTimeFromUTC(posPrevious.timestamp));

            var dst=posPrevious.latlng.distanceTo(posCurrent.latlng);
            if ((posCurrent.altitude=="NA") || (posPrevious.altitude=="NA")) {
                var alt="NA";
            } else {
                var alt=posCurrent.altitude-posPrevious.altitude
            }
            var tm=(posCurrent.timestamp-posPrevious.timestamp)/1000
            var bng=L.GeometryUtil.bearing(posPrevious.latlng, posCurrent.latlng)
            if (alt=="NA") {
                var clr="NA";
            } else {
                var clr=(alt/tm*60*60).toFixed(1);
            }
            $("#info_dif_dst").val(dst.toFixed(1));
            $("#info_dif_alt").val(alt);
            $("#info_dif_tm").val(tm.toFixed(1));
            $("#info_dif_bng").val(bng.toFixed(1));
            $("#info_dif_vel").val(((dst/tm*60*60)/1000).toFixed(3));
            $("#info_dif_clr").val(clr);
        }

        posPrevious=posCurrent;
    }
}

function populatePoints(){
    var lyrPrevious;
    var dst;
    var dstSum=0;
    var bng;
    var start;
    var strPopup;
    var tm;
    var strTable="<table class='table'><tr class='table-header'><th>ID</th><th>Time</th><th>Dist (m)</th><th>Bearing</th><th></th></tr>";
    lyrBreadcrumbs.eachLayer(function(lyr){
        if (lyrPrevious) {
            strPopup=lyr.getPopup().getContent();
            start = strPopup.indexOf("Time: ")+6;
            tm=strPopup.substring(start, start+8);
            dst = lyrPrevious.getLatLng().distanceTo(lyr.getLatLng());
            dstSum+=dst;
            bng = L.GeometryUtil.bearing(lyrPrevious.getLatLng(), lyr.getLatLng())
            strTable+="<tr><td>"+L.stamp(lyr)+"</td><td>"+tm+"</td><td>"+dst.toFixed(1)+"</td><td>"+bng.toFixed(0)+"</td><td><span class='btnFindPt' data-id='"+L.stamp(lyr)+"'><i class='fa fa-search'</i></span></td></tr>";
            lyrPrevious=lyr;
        } else {
            strPopup=lyr.getPopup().getContent();
            start = strPopup.indexOf("Time: ")+6;
            tm=strPopup.substring(start, start+8);
            strTable+="<tr><td>"+L.stamp(lyr)+"</td><td>"+tm+"</td><td>NA</td><td>NA</td><td><span class='btnFindPt' data-id='"+L.stamp(lyr)+"'><i class='fa fa-search'</i></span></td></tr>";
            lyrPrevious=lyr;
        }
    });
    strTable+="<tr class='table-header'><th>Total</th><th></th><th>"+dstSum.toFixed(0)+"</th><th></th><th></th></tr>";
    strTable+="</table>";
    strTable+="<button id='btnClearCrumbs' class='btn btn-danger btn-block btn-no-top-margin'>Clear Breadcrumbs</button>";
    $("#points").html(strTable);
    $(".btnFindPt").click(function(){
        var id = $(this).attr("data-id");
        var ll = lyrBreadcrumbs.getLayer(id).getLatLng();
        mymap.setView(ll,17);
        openSubScreen();
    });
    $("#btnClearCrumbs").click(function(){
        if (confirm("Are you sure you want delete all the crumbs?")) {
            lyrBreadcrumbs.clearLayers();
            populatePoints();
        }
    });
}

function storeSettings(){
    var jsnSettings={};
    jsnSettings.autolocate=$("#btnAutolocate").html();
    jsnSettings.numAutolocate=$("#numAutolocate").val();
    jsnSettings.breadcrumbs=$("#btnBreadcrumbs").html();
    jsnSettings.numBreadcrumbs=$("#numBreadcrumbs").val();
    jsnSettings.filter=$("#btnFilter").html();
    jsnSettings.numFilter=$("#numFilter").val();
    localStorage.jsnSettings=JSON.stringify(jsnSettings);
}