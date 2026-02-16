package com.example.networkcollector

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.net.ConnectivityManager
import android.net.wifi.WifiManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.android.gms.location.LocationServices
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private val fusedLocationClient by lazy { LocationServices.getFusedLocationProviderClient(this) }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { updateStatus("Permissions updated. Tap Send Telemetry.") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        val sendButton = findViewById<Button>(R.id.sendTelemetryButton)

        sendButton.setOnClickListener {
            requestPermissionsIfNeeded()
            sendTelemetryAndPredict()
        }
    }

    private fun requestPermissionsIfNeeded() {
        permissionLauncher.launch(
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION,
                Manifest.permission.ACCESS_WIFI_STATE,
                Manifest.permission.ACCESS_NETWORK_STATE
            )
        )
    }

    private fun sendTelemetryAndPredict() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val payload = buildPayload()
                val response = postJson("http://YOUR_SERVER_IP:8000/predict", payload)
                withContext(Dispatchers.Main) { updateStatus("Prediction response: $response") }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { updateStatus("Error: ${e.message}") }
            }
        }
    }

    private fun buildPayload(): JSONObject {
        val wifiManager = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        val connectivity = applicationContext.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val activeNetwork = connectivity.activeNetwork
        val caps = connectivity.getNetworkCapabilities(activeNetwork)

        val wifiInfo = wifiManager.connectionInfo
        val location = getLastLocation()

        val payload = JSONObject()
        payload.put("device_id", android.os.Build.MODEL)
        payload.put("network_type", "WiFi")
        payload.put("latitude", location?.latitude)
        payload.put("longitude", location?.longitude)
        payload.put("signal_strength_dbm", wifiInfo.rssi)
        payload.put("downlink_mbps", caps?.linkDownstreamBandwidthKbps?.div(1000.0))
        payload.put("rtt_ms", JSONObject.NULL)
        payload.put("latency_ms", JSONObject.NULL)
        payload.put("packet_loss_pct", JSONObject.NULL)
        payload.put("store_telemetry", true)
        return payload
    }

    private fun getLastLocation(): Location? {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED
        ) {
            return null
        }
        return try {
            val task = fusedLocationClient.lastLocation
            while (!task.isComplete) {
                Thread.sleep(30)
            }
            task.result
        } catch (_: Exception) {
            null
        }
    }

    private fun postJson(url: String, body: JSONObject): String {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.requestMethod = "POST"
        conn.setRequestProperty("Content-Type", "application/json")
        conn.doOutput = true

        OutputStreamWriter(conn.outputStream).use { it.write(body.toString()) }
        val stream = if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
        return stream.bufferedReader().use { it.readText() }
    }

    private fun updateStatus(msg: String) {
        statusText.text = msg
    }
}
